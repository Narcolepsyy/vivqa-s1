import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from builders.dataset_builder import build_dataset
from builders.model_builder import build_model
from builders.task_builder import META_TASK
from builders.vocab_builder import build_vocab
from data_utils.datasets.s1_dataset import s1_collate
from utils import viso_calibration as calib
from utils import viso_rlcd as rlcd
from utils.logging_utils import setup_logger
from utils.viso_s1 import label_space, read_jsonl

import utils.viso_s1  # noqa: F401  (s1bench on sys.path)
from s1bench.metrics import evaluate as s1_evaluate

logger = setup_logger()
SPLITS = ("train", "calib", "dev", "test")


class IncompleteRun(SystemExit):
    """Raised (exit code 3) when a run stops on purpose with resumable state saved in `last.pth`
    (deadline reached or TRAINING.STOP_AFTER_EPOCHS)."""

    def __init__(self, message=""):
        super().__init__(3)
        self.message = message


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preds_dict(items, probs):
    """{id: {"probabilities": {label key: p}}} in the format s1bench.metrics.evaluate expects."""
    return {it["id"]: {"probabilities": {k: float(probs[i][j]) for j, k in enumerate(label_space(it["questions"]["q"]))}}
            for i, it in enumerate(items)}


def worker_init(_worker_id):
    """DataLoader workers: one math thread each (N workers x all cores oversubscribe the CPU and starve the GPU)"""
    torch.set_num_threads(1)


def loader_kwargs(workers, pin=False):
    return dict(num_workers=workers, worker_init_fn=worker_init if workers else None, persistent_workers=False, pin_memory=bool(pin) and torch.cuda.is_available()) if workers else {}


@torch.no_grad()
def collect_logits(model, dataset, batch_size, device, workers=0):
    """Raw (temperature-free) masked logits of the typed layout for a whole split, in dataset order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=s1_collate, **loader_kwargs(workers))
    ids, types, logits, masks = [], [], [], []
    for batch in loader:
        batch.to(device)
        out = model(batch)["logits"].float().cpu()
        ids.append(batch.id.cpu().numpy()); types.append(batch.type_id.cpu().numpy())
        logits.append(out.numpy()); masks.append(batch.space_mask.cpu().numpy())
    return dict(ids=np.concatenate(ids), type_id=np.concatenate(types), logits=np.concatenate(logits),
                space_mask=np.concatenate(masks))


@META_TASK.register()
class TypedDecisionTask:
    """Trains / evaluates ViSO (typed heads) and FlatS1 (baseline A0) on ViVQA-S1.

    Objectives (LOSS.NAME): hard_ce | soft_ce | rlcd. rlcd = stage 1 soft_ce (best dev NLL, early stopping) then
    stage 2 `EPOCHS_RL` epochs of soft_ce + PG_WEIGHT * policy-gradient (utils/viso_rlcd.py). FlatS1 uses the flat
    answer vocabulary: hard_ce / soft_ce over the full flat logits (rlcd is not defined for it). Model selection for
    every method: best DEV primary NLL (s1bench.metrics on temperature-free probabilities).
    The class does not inherit BaseTask (its constructor builds an Adam + NLLLoss setup and uses cuda RNG states)."""

    def __init__(self, config):
        self.config = config
        tr = config.TRAINING
        seed_everything(tr.SEED)
        torch.backends.cudnn.deterministic = True      # GPU runs are still not bit-exact (documented); CPU is the reference
        torch.backends.cudnn.benchmark = False
        self.resume = bool(getattr(tr, "RESUME", False))
        self.deadline = float(getattr(tr, "DEADLINE_UNIX", 0.0))                 # 0 = none
        self.stop_after = int(getattr(tr, "STOP_AFTER_EPOCHS", 0))               # 0 = off (tests: simulated interruption)
        self.session_epoch_seconds = []
        self.t_start = time.time()
        self.device = torch.device(config.MODEL.DEVICE)
        self.flat = config.MODEL.ARCHITECTURE == "FlatS1"
        self.checkpoint_path = os.path.join(tr.CHECKPOINT_PATH, config.MODEL.NAME)
        self.output_path = os.path.join(tr.OUTPUT_PATH, config.MODEL.NAME)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        vocab_file = os.path.join(self.checkpoint_path, "vocab.bin")
        if os.path.isfile(vocab_file):
            self.vocab = pickle.load(open(vocab_file, "rb"))
        else:
            self.vocab = build_vocab(config.DATASET.VOCAB)
            pickle.dump(self.vocab, open(vocab_file, "wb"))

        fd, s1 = config.DATASET.FEATURE_DATASET, config.DATASET.VOCAB.S1_DIR
        self.datasets = {n: build_dataset(os.path.join(s1, f"{n}.jsonl"), self.vocab, fd) for n in SPLITS}
        self.batch_size = fd.BATCH_SIZE
        self.train_gen = torch.Generator().manual_seed(tr.SEED)
        self.train_loader = DataLoader(self.datasets["train"], batch_size=self.batch_size, shuffle=True, collate_fn=s1_collate, generator=self.train_gen,
                                       **({"num_workers": fd.WORKERS, "worker_init_fn": worker_init, "pin_memory": bool(getattr(fd, "PIN_MEMORY", False)) and torch.cuda.is_available(), "persistent_workers": False}
                                          if fd.WORKERS else {"num_workers": 0}))
        self.rl_gen = torch.Generator().manual_seed(tr.SEED + 1)          # RLCD noise has its own generator

        self.model = build_model(config.MODEL, self.vocab)
        self.optim = self._make_optimizer()
        self.global_step = 0
        self.log_file = os.path.join(self.output_path, "train_log.jsonl")
        with open(os.path.join(self.output_path, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(config.dump())

    # ------------------------------------------------------------------ setup
    def _make_optimizer(self):
        tr = self.config.TRAINING
        # VISION_PATH_AS_HEADS: vision_embedding / vision_proj / vision_attr_reduce / box_proj / feature_norm / prod_v use LR_HEADS (the vision side got ~10x smaller gradients than the text side);
        # TEXT_PROJ_AS_HEADS: the Linear(768 -> 512) over the cached BERT states (CachedBertEmbedding) uses LR_HEADS. Both default False (every parameter of the encoder uses LR_ENCODER).
        to_heads = []
        if getattr(tr, "VISION_PATH_AS_HEADS", False):
            to_heads += ["vision_embedding.", "vision_proj.", "vision_attr_reduce.", "box_proj.", "feature_norm.", "prod_v."]
        if getattr(tr, "TEXT_PROJ_AS_HEADS", False):
            to_heads += ["text_embedding.proj."]
        if getattr(tr, "ENCODER_AS_HEADS", False):                  # one learning rate (LR_HEADS) for every parameter: from-scratch models without a pretrained part (B-BUTD)
            return torch.optim.AdamW([{"params": list(self.model.parameters()), "lr": tr.LR_HEADS}], weight_decay=tr.WEIGHT_DECAY)
        ft = list(self.model.finetune_parameters()) if hasattr(self.model, "finetune_parameters") else []     # ViSO-FT: the unfrozen parameters of the live towers get their own LR_FT
        ft_ids = {id(p) for p in ft}
        enc = [p for n, p in self.model.encoder.named_parameters() if not n.startswith("classify.") and not any(n.startswith(t) for t in to_heads) and id(p) not in ft_ids]
        seen = {id(p) for p in enc} | ft_ids
        heads = [p for p in self.model.parameters() if id(p) not in seen and (p.requires_grad or not ft_ids)]       # typed heads / flat classifier (+ the groups above); frozen tower weights excluded
        groups = [{"params": enc, "lr": tr.LR_ENCODER}, {"params": heads, "lr": tr.LR_HEADS}]
        if ft:
            groups.append({"params": ft, "lr": float(getattr(tr, "LR_FT", 1e-5))})
        return torch.optim.AdamW(groups, weight_decay=tr.WEIGHT_DECAY)

    def _make_scheduler(self, total_steps):
        warm = max(1, int(self.config.TRAINING.WARMUP_STEPS))

        def lam(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total_steps - warm))))

        return torch.optim.lr_scheduler.LambdaLR(self.optim, lam)

    # ------------------------------------------------------------------ losses
    def _flat_targets(self, out, batch, soft):
        """Flat-vocab targets for A0: soft mass scattered to the flat indices of the item's labels (labels absent
        from the flat vocab are dropped and the rest renormalised; items with no mass left get weight 0)."""
        n_flat = out["flat_logits"].shape[1]
        idx = batch.flat_index
        valid = (idx >= 0) & batch.space_mask
        if soft:
            w = batch.target * valid
        else:
            w = torch.zeros_like(batch.target)
            w.scatter_(1, batch.gold_idx[:, None], 1.0)
            w = w * valid
        tgt = torch.zeros(idx.shape[0], n_flat, device=idx.device).scatter_add_(1, idx.clamp(min=0), w)
        mass = tgt.sum(1)
        keep = mass > 0
        return tgt / mass.clamp(min=1e-12)[:, None], keep

    def item_loss(self, out, batch, name, pg_info=None, sigma=None):
        """Per-item loss (B,) for the objective `name`."""
        if self.flat and name == "rlcd":
            raise ValueError("RLCD is defined for the typed model (ViSO) only")
        lg, mask = out["logits"], batch.space_mask
        if self.flat:
            tgt, keep = self._flat_targets(out, batch, soft=(name == "soft_ce"))
            logp = torch.log_softmax(out["flat_logits"], dim=-1)
            return -(tgt * logp).sum(-1) * keep
        if name == "hard_ce":
            return rlcd.hard_ce(lg, mask, batch.gold_idx)
        loss = rlcd.soft_ce(lg, mask, batch.target)
        kd = float(getattr(self.config.LOSS, "KD_WEIGHT", 0.0))
        if kd > 0 and "teacher_target" in batch:                          # knowledge distillation: (1 - a) CE(human soft) + a CE(teacher soft); items the teacher refused keep the human loss only
            loss = torch.where(batch.has_teacher, (1.0 - kd) * loss + kd * rlcd.soft_ce(lg, mask, batch.teacher_target), loss)
        if name == "rlcd":
            lc = self.config.LOSS
            pg, mean_r = rlcd.policy_gradient_loss(lg, mask, batch.target, batch.type_id, sigma, self.rl_gen,
                                                   lc.NUM_SAMPLES, lc.SPHERICAL_WEIGHT, lc.RPS_WEIGHT)
            pg_scale = float(sigma) if (bool(getattr(lc, "PG_SIGMA_SCALE", False)) and sigma is not None) else 1.0   # the PG gradient grows like 1/sigma (audit: scripts/viso_rl_gradcheck.py)
            loss = loss + lc.PG_WEIGHT * pg_scale * pg
            if pg_info is not None:
                pg_info.append(float(mean_r))
        return loss

    def weighted(self, per_item, type_id):
        """per-item loss times LOSS.TYPE_WEIGHTS[type] (training only; dev / selection losses stay unweighted). [1, 1, 1] returns per_item unchanged."""
        w = list(getattr(self.config.LOSS, "TYPE_WEIGHTS", [1.0, 1.0, 1.0]))
        if all(float(x) == 1.0 for x in w):
            return per_item
        return per_item * torch.tensor([float(x) for x in w], device=per_item.device)[type_id]

    # ------------------------------------------------------------------ training
    def train_epoch(self, loss_name, scheduler, sigma_fn=None, total_steps=1, step0=0):
        self.model.train()
        tr, balance = self.config.TRAINING, self.config.LOSS.TYPE_BALANCE
        running, n, rewards = 0.0, 0, []
        by_type = {t: [0.0, 0] for t in ("noul", "score", "choice")}           # per-type train loss of the epoch (logged as train_loss_<type>)
        for it, batch in enumerate(self.train_loader):
            batch.to(self.device)
            sigma = sigma_fn((step0 + it) / max(1, total_steps - 1)) if sigma_fn else None
            out = self.model(batch)
            per_item = self.item_loss(out, batch, loss_name, rewards, sigma)
            loss = rlcd.reduce_loss(self.weighted(per_item, batch.type_id), batch.type_id, balance)
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tr.GRAD_CLIP)
            self.optim.step()
            scheduler.step()
            self.global_step += 1
            running += float(loss.detach()) * batch.id.shape[0]
            n += batch.id.shape[0]
            for t, name in enumerate(("noul", "score", "choice")):
                k = batch.type_id == t
                if k.any():
                    by_type[name][0] += float(per_item.detach()[k].sum())
                    by_type[name][1] += int(k.sum())
        self.last_train_by_type = {f"train_loss_{t}": round(a / b, 6) for t, (a, b) in by_type.items() if b}
        return running / max(n, 1), (float(np.mean(rewards)) if rewards else None)

    @torch.no_grad()
    def mean_loss(self, split, loss_name="soft_ce"):
        """Mean per-item loss over a split, eval mode (no noise; rlcd is not used here)."""
        self.model.eval()
        loader = DataLoader(self.datasets[split], batch_size=self.batch_size, shuffle=False, collate_fn=s1_collate)
        tot, n = 0.0, 0
        for batch in loader:
            batch.to(self.device)
            tot += float(self.item_loss(self.model(batch), batch, loss_name).sum())
            n += batch.id.shape[0]
        return tot / max(n, 1)

    def evaluate_split(self, split):
        """s1bench metrics (temperature-free) for a split; primary 'all' + per type + S1-Human."""
        ds = self.datasets[split]
        lg = collect_logits(self.model, ds, self.batch_size, self.device, int(getattr(self.config.DATASET.FEATURE_DATASET, "EVAL_WORKERS", 0)))
        probs = calib.probs_from_logits(lg["logits"], lg["space_mask"])
        return s1_evaluate(ds.items, preds_dict(ds.items, probs))

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, name, **extra):
        torch.save({"state_dict": self.model.state_dict(), **extra}, os.path.join(self.checkpoint_path, name))

    def load_checkpoint(self, name="best_model.pth"):
        path = os.path.join(self.checkpoint_path, name)
        if not os.path.isfile(path):
            return None
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ck["state_dict"])
        return ck

    def _log(self, **row):
        logger.info(json.dumps(row, ensure_ascii=False))
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ resume / deadline
    def _rng_state(self):
        st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
              "train_gen": self.train_gen.get_state(), "rl_gen": self.rl_gen.get_state()}
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
        return st

    def _set_rng_state(self, st):
        torch.set_rng_state(st["torch"])
        np.random.set_state(st["numpy"])
        random.setstate(st["python"])
        self.train_gen.set_state(st["train_gen"])
        self.rl_gen.set_state(st["rl_gen"])
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])

    def _save_last(self, stage, epoch, best, best_epoch, bad, stage_finished, scheduler, results):
        """last.pth after every epoch: everything needed to continue exactly where the run stopped."""
        state = {"model": self.model.state_dict(), "optim": self.optim.state_dict(), "scheduler": scheduler.state_dict(),
                 "stage": stage, "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad": bad,
                 "stage_finished": stage_finished, "global_step": self.global_step, "results": results,
                 "wall_seconds": self.wall_before + time.time() - self.t_start, "rng": self._rng_state()}
        tmp = os.path.join(self.checkpoint_path, "last.pth.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(self.checkpoint_path, "last.pth"))      # atomic: a kill never leaves a torn file

    def _load_last(self):
        path = os.path.join(self.checkpoint_path, "last.pth")
        if not (self.resume and os.path.isfile(path)):
            return None
        # our own file (numpy RNG tuples); CPU map_location because the RNG states must stay CPU ByteTensors
        return torch.load(path, map_location="cpu", weights_only=False)

    def _maybe_stop(self, epoch_seconds, work_remains):
        """Deadline / simulated-interruption exit (code 3): last.pth has just been written."""
        if not work_remains:
            return
        self.session_epoch_seconds.append(epoch_seconds)
        if self.stop_after and len(self.session_epoch_seconds) >= self.stop_after:
            raise IncompleteRun(f"STOP_AFTER_EPOCHS={self.stop_after}")
        if self.deadline > 0:
            mean = sum(self.session_epoch_seconds) / len(self.session_epoch_seconds)
            if time.time() + 1.2 * mean > self.deadline:
                raise IncompleteRun("deadline")

    def _run_stage(self, stage, loss_name, epochs, patience, best_name, sigma_fn=None, is_last_stage=True,
                   resume_state=None, results=None):
        steps_per_epoch = len(self.train_loader)
        total = epochs * steps_per_epoch
        scheduler = self._make_scheduler(total)
        best, best_epoch, bad, start_epoch = float("inf"), 0, 0, 1
        if resume_state is not None:                  # continue inside this stage
            self.model.load_state_dict(resume_state["model"])
            self.optim.load_state_dict(resume_state["optim"])
            scheduler.load_state_dict(resume_state["scheduler"])
            best, best_epoch, bad = resume_state["best"], resume_state["best_epoch"], resume_state["bad"]
            start_epoch = resume_state["epoch"] + 1
            self.global_step = resume_state["global_step"]
            self._set_rng_state(resume_state["rng"])
            logger.info("resumed stage %s after epoch %d", stage, resume_state["epoch"])
        early = False
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            loss, reward = self.train_epoch(loss_name, scheduler, sigma_fn, total, (epoch - 1) * steps_per_epoch)
            ev = self.evaluate_split("dev")
            p = ev["primary"]["all"]
            row = dict(stage=stage, loss=loss_name, epoch=epoch, train_loss=round(loss, 6), mean_reward=reward, seconds=round(time.time() - t0, 1),
                       peak_gpu_gb=(round(torch.cuda.max_memory_allocated(self.device) / 2 ** 30, 2) if self.device.type == "cuda" else None),
                       dev_acc=round(p["accuracy"], 6), dev_brier=round(p["brier"], 6), dev_nll=round(p["nll"], 6),
                       dev_ece=round(p["ece"], 6), dev_aurc=round(p["aurc"], 6),
                       **getattr(self, "last_train_by_type", {}),
                       **{f"dev_acc_{t}": round(ev["primary"][t]["accuracy"], 6) for t in ("noul", "score", "choice") if t in ev["primary"]})
            self._log(**row)
            if p["nll"] < best:
                best, best_epoch, bad = p["nll"], epoch, 0
                self.save_checkpoint(best_name, epoch=epoch, stage=stage, dev_nll=p["nll"])
            else:
                bad += 1
                if patience and bad >= patience:
                    self._log(stage=stage, event="early_stop", epoch=epoch)
                    early = True
            stage_ends = early or epoch == epochs
            self._save_last(stage, epoch, best, best_epoch, bad, stage_ends, scheduler, results or [])
            if early:
                break
            self._maybe_stop(time.time() - t0, work_remains=(not stage_ends) or (not is_last_stage))
        self.load_checkpoint(best_name)
        return {"stage": stage, "loss": loss_name, "best_dev_nll": best, "best_epoch": best_epoch,
                "epochs_run": epoch if epochs >= start_epoch else resume_state["epoch"]}

    def _plan(self):
        """[(stage, loss_name, epochs, patience, best_name, sigma_fn)] for the configured objective."""
        tr, lc = self.config.TRAINING, self.config.LOSS
        if lc.NAME in ("hard_ce", "soft_ce"):
            return [("sup", lc.NAME, tr.EPOCHS_SUP, tr.PATIENCE, "best_model.pth", None)]
        if lc.NAME != "rlcd":
            raise ValueError(f"unknown LOSS.NAME {lc.NAME}")
        sig = lambda f: rlcd.sigma_at(f, lc.SIGMA_START, lc.SIGMA_END)
        sup = ("sup", "soft_ce", tr.EPOCHS_SUP, tr.PATIENCE, "best_model_stage1.pth", None)
        rl = ("rl", "rlcd", tr.EPOCHS_RL, 0, "best_model.pth", sig)
        return {"all": [sup, rl], "sup": [sup], "rl": [rl]}[lc.STAGE]

    def start(self):
        """Train (resumable). Returns normally when the run is finished (DONE.json written, last.pth deleted);
        raises IncompleteRun (exit code 3) when it stopped on purpose with resumable state."""
        done_file = os.path.join(self.checkpoint_path, "DONE.json")
        if self.resume and os.path.isfile(done_file):
            logger.info("DONE.json exists: skipping %s", self.checkpoint_path)
            return json.load(open(done_file, encoding="utf-8"))
        if not self.resume:                              # a fresh run never inherits stale resume state
            for f in ("DONE.json", "last.pth"):
                if os.path.isfile(os.path.join(self.checkpoint_path, f)):
                    os.remove(os.path.join(self.checkpoint_path, f))
        last = self._load_last()
        self.wall_before = last["wall_seconds"] if last else 0.0
        plan, results = self._plan(), (list(last["results"]) if last else [])
        for k, (stage, loss_name, epochs, patience, best_name, sigma_fn) in enumerate(plan):
            is_last = k == len(plan) - 1
            if last is not None and last["stage"] != stage and any(r["stage"] == stage for r in results):
                continue                                 # this stage finished in an earlier session
            resume_state = last if (last is not None and last["stage"] == stage) else None
            if resume_state is not None and resume_state["stage_finished"]:
                self.load_checkpoint(best_name)           # finished stage: restore its best model, RNG streams and step
                self._set_rng_state(resume_state["rng"])
                self.global_step = resume_state["global_step"]
                if stage == "sup" and loss_name == "soft_ce" and best_name == "best_model_stage1.pth":
                    self.save_checkpoint("best_model.pth", note="stage-1 best")
                results.append({"stage": stage, "loss": loss_name, "best_dev_nll": resume_state["best"],
                                "best_epoch": resume_state["best_epoch"], "epochs_run": resume_state["epoch"]})
                last = None
                continue
            if stage == "rl":
                if resume_state is None:
                    if self.config.LOSS.STAGE == "rl" and self.load_checkpoint("best_model_stage1.pth") is None:
                        raise FileNotFoundError("LOSS.STAGE=rl needs best_model_stage1.pth from a previous sup stage")
                    self.optim = self._make_optimizer()   # fresh optimiser state for stage 2
                else:
                    self.optim = self._make_optimizer()   # same structure as the saved one; state loaded in _run_stage
            res = self._run_stage(stage, loss_name, epochs, patience, best_name, sigma_fn, is_last, resume_state, results)
            results.append(res)
            last = None
            if stage == "sup" and best_name == "best_model_stage1.pth":
                self.save_checkpoint("best_model.pth", note="stage-1 best")
        final = results[-1]
        done = {"loss": self.config.LOSS.NAME, "stage_results": results, "best_epoch": final["best_epoch"],
                "best_dev_nll": final["best_dev_nll"], "wall_seconds": round(self.wall_before + time.time() - self.t_start, 1),
                "completed_unix": time.time()}
        with open(done_file, "w", encoding="utf-8") as f:
            json.dump(done, f, indent=1)
        if os.path.isfile(os.path.join(self.checkpoint_path, "last.pth")):
            os.remove(os.path.join(self.checkpoint_path, "last.pth"))
        return done

    def get_predictions(self):
        """Dump temperature-free logits of the selected model for calib / dev / test (npz) -> output dir."""
        self.load_checkpoint("best_model.pth")
        for split in ("calib", "dev", "test"):
            lg = collect_logits(self.model, self.datasets[split], self.batch_size, self.device, int(getattr(self.config.DATASET.FEATURE_DATASET, "EVAL_WORKERS", 0)))
            calib.save_logits(os.path.join(self.output_path, f"logits_{split}.npz"), **lg)
        logger.info("logits written to %s", self.output_path)
