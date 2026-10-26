"""LoRA decision fine-tune of a VLM in SYSTEM ONE mode (the trained row after zero-shot and head-only in scripts/s1_vlm_eval.py).
Same interface as the head-only baseline: the shared Vietnamese lettered prompt, ONE forward pass, the hidden state at the answer position -> a 16 x d letter head initialised from the unembedding rows of the
letters A..P (so it equals zero-shot at step 0). Here the head is trained JOINTLY with LoRA adapters; LoRA is the only difference to head-only.
  * LoRA r=16, alpha=16, dropout 0 on the LANGUAGE model's attention (full attention q/k/v/o, linear attention in_proj_qkv / in_proj_z / out_proj) and MLP (gate / up / down) projections. The vision tower is FROZEN.
  * LR 2e-4 (LoRA) / 1e-4 (head), AdamW weight decay 0.01, cosine schedule, 10 warm-up steps, grad clip 1.0; effective batch 32 (8 x 4); 1 epoch over the full train split; bf16.
  * loss = soft cross-entropy on the human soft labels (like ViSO); for CHOICE items the option order is randomly permuted per item at training time (noul / score keep their natural order, which is the only
    order they are evaluated in; --permute-all permutes them too).
  * dev NLL on a seeded type-stratified dev subset at step 0 and every 1/4 of the epoch; the best-dev-NLL state is kept; final predictions on the FULL calib / dev / test (choice items in 2 orders, averaged),
    per-type temperature on calib, same outputs / views as the other VLM baselines (finish_run -> evaluate.py: Image-unseen headline, equivalence-aware).
  * --mode txt: the identical recipe with NO image in the prompt (the language-shortcut ceiling); the prompt text is unchanged.
Output (--model-dir): meta.json, adapter/ (PEFT) + head.pt (+ best_state.pt), lora_s<seed>/ {train_log.json, logits_*.npz, calibration/, views.json}, summary (written by the caller: run.sh)."""
import argparse
import json
import math
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIVQA_S1_ROOT", os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2")))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")                  # tokenizers run inside DataLoader workers
import numpy as np  # noqa: E402
import torch  # noqa: E402

import s1_vlm_eval as V  # noqa: E402
from utils import viso_calibration as C  # noqa: E402
from utils.atomic import atomic_dir, atomic_json, atomic_torch_save  # noqa: E402
from utils.viso_s1 import MAX_SPACE, label_space  # noqa: E402

LORA_TARGETS = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def prompt_perm(it, perm):
    q = it["questions"]["q"]
    texts = V.option_texts(q)
    return V.PROMPT.format(q=q["instructions"], options="\n".join(f"{V.LETTERS[k]}. {texts[j]}" for k, j in enumerate(perm)))


def target_perm(it, perm):
    """soft labels in LETTER order for the option permutation `perm` (letter k shows label perm[k]) and the valid-letter mask"""
    sp = label_space(it["questions"]["q"])
    t = np.zeros(MAX_SPACE, np.float32)
    m = np.zeros(MAX_SPACE, bool)
    m[:len(sp)] = True
    for k, j in enumerate(perm):
        t[k] = it["gold"]["q"]["probabilities"].get(sp[j], 0.0)
    return t, m


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, items, store, seed, permute_all, use_image):
        self.items, self.store, self.seed, self.permute_all, self.use_image = items, store, seed, permute_all, use_image

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        q = it["questions"]["q"]
        n = len(V.option_texts(q))
        if q["type"] == "choice" or self.permute_all:
            perm = [int(x) for x in np.random.RandomState(self.seed * 1000003 + i).permutation(n)]
        else:
            perm = list(range(n))
        t, m = target_perm(it, perm)
        return prompt_perm(it, perm), (self.store.open(it["coco_id"]) if self.use_image else None), t, m


class EvalSet(torch.utils.data.Dataset):
    def __init__(self, items, units, store, use_image):
        self.items, self.units, self.store, self.use_image = items, units, store, use_image

    def __len__(self):
        return len(self.units)

    def __getitem__(self, k):
        i, order = self.units[k]
        it = self.items[i]
        return it["id"], order, V.build_prompt(it, order), (self.store.open(it["coco_id"]) if self.use_image else None)


def make_inputs(proc, prompts, images, use_image):
    texts = []
    for p in prompts:
        content = ([{"type": "image"}] if use_image else []) + [{"type": "text", "text": p}]
        texts.append(proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False, enable_thinking=False))
    return proc(text=texts, images=images, return_tensors="pt", padding=True) if use_image else proc(text=texts, return_tensors="pt", padding=True)


def dev_nll(items, units, letter_logits):
    """mean NLL of the hard gold label under the order-averaged label distribution (the head-only selection metric)"""
    gold = C.gold_arrays(items)["gold_idx"]
    nll = 0.0
    for i, it in enumerate(items):
        n = len(label_space(it["questions"]["q"]))
        orders = (1, 2) if it["questions"]["q"]["type"] == "choice" else (1,)
        z = V.to_label_logits(it, {o: letter_logits[(it["id"], o)] for o in orders})[:n]
        p = np.exp(z - z.max())
        p /= p.sum()
        nll -= np.log(max(p[gold[i]], 1e-12))
    return float(nll / max(len(items), 1))


RESUME_KEYS = ("mode", "model", "revision", "seed", "epochs", "batch", "accum", "lr_lora", "lr_head", "wd", "warmup", "rank", "alpha", "max_len", "max_pixels", "limit", "max_steps", "eval_items", "eval_points", "permute_all")


def resume_signature(a):
    return {k: getattr(a, k) for k in RESUME_KEYS}


def save_resume(out, payload):
    """atomic: <out>/resume/state.pt (+ state_meta.json): everything needed to continue at a step boundary with the same data order"""
    d = os.path.join(out, "resume")
    os.makedirs(d, exist_ok=True)
    atomic_torch_save(payload, os.path.join(d, "state.pt"))
    atomic_json(os.path.join(d, "state_meta.json"), {"step": payload["step"], "seen": payload["seen"], "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "best_step": payload["best"]["step"], "best_dev_nll": payload["best"]["nll"]})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["img", "txt"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--licence", default="")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--s1-data", required=True)
    ap.add_argument("--root", required=True, help="benchmark root (evaluate.py)")
    ap.add_argument("--image-roots", nargs="*", default=[])
    ap.add_argument("--image-sha", default=os.path.join(REPO, "kaggle", "s1_images_sha256.tsv.gz"))
    ap.add_argument("--image-cache", default=None)
    ap.add_argument("--download-images", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--eval-items", type=int, default=1500, help="size of the seeded, type-stratified dev subset used for the periodic dev NLL (final predictions use the full splits)")
    ap.add_argument("--eval-points", type=int, default=4, help="dev evaluations per epoch (+ one at step 0)")
    ap.add_argument("--permute-all", action="store_true", help="permute the options of noul / score items too (default: choice only)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items of every split")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after this many optimizer steps (the schedule is scaled to it)")
    ap.add_argument("--watchdog-sec", type=float, default=900.0)
    ap.add_argument("--resume", action="store_true", help="continue from <model-dir>/resume/state.pt (written at every dev evaluation: adapter, head, optimizer, scheduler, step, best state, history) if it exists and was made with the same settings")
    ap.add_argument("--stop-after-step", type=int, default=0, help="testing: exit (rc 0) right after the checkpoint written at this optimizer step")
    ap.add_argument("--save-zeroshot", action="store_true", help="also evaluate and save <model-dir>/zeroshot/ at step 0 before LoRA updates")
    ap.add_argument("--zeroshot-debias", action="store_true", help="with --save-zeroshot: also save <model-dir>/zeroshot_debiased/ (calib-mean letter-position bias subtracted before the log-softmax, as scripts/s1_vlm_eval.py --debias-letters) and the raw letter logits (zeroshot_letters_<split>.npz)")
    ap.add_argument("--zeroshot-only", action="store_true", help="exit right after the zero-shot evaluation(s) (no dev eval, no training)")
    a = ap.parse_args(argv)
    a.s1_data, a.root = (os.path.abspath(os.path.expanduser(p)) for p in (a.s1_data, a.root))
    a.image_roots = [os.path.abspath(os.path.expanduser(p)) for p in a.image_roots]
    import faulthandler
    faulthandler.enable()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    V.require_fast_fla(a.device)
    use_image = a.mode == "img"
    out = a.model_dir
    run = os.path.join(out, f"lora_s{a.seed}")
    os.makedirs(run, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    t_begin = time.time()

    hf_cfg = AutoConfig.from_pretrained(a.model, revision=a.revision, trust_remote_code=False)
    is_d3 = getattr(hf_cfg, "architectures", None) == ["Qwen3_5Model"]
    if is_d3:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        model = Qwen3_5Model.from_pretrained(a.model, revision=a.revision, dtype=dtype, attn_implementation="sdpa", trust_remote_code=False).to(a.device)
        dcfg = json.load(open(hf_hub_download(a.model, "decision_config.json", revision=a.revision)))
        use_noncausal = dcfg.get("attention_mode") == "noncausal_full_attention"
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, revision=a.revision, dtype=dtype, trust_remote_code=False).to(a.device)
        use_noncausal = False
    proc = AutoProcessor.from_pretrained(a.model, revision=a.revision)
    proc.tokenizer.padding_side = "left"
    if use_image and a.max_pixels:
        try:
            proc.image_processor.size = {"shortest_edge": a.min_pixels, "longest_edge": a.max_pixels}
        except Exception as e:
            print("could not set max_pixels:", e)
    letter_ids = []
    for L in V.LETTERS:
        ids = proc.tokenizer.encode(L, add_special_tokens=False)
        assert len(ids) == 1, (L, ids)
        letter_ids.append(ids[0])
    if is_d3:
        ro_w = load_file(hf_hub_download(a.model, "readout.safetensors", revision=a.revision))["weight"]
        assert dcfg["token_ids"][:len(letter_ids)] == letter_ids, (dcfg["token_ids"][:len(letter_ids)], letter_ids)
        W0 = ro_w[:len(letter_ids)].detach().float().clone()
        n_params = sum(p.numel() for p in model.parameters()) + ro_w.numel()
    else:
        W0 = model.get_output_embeddings().weight[letter_ids].detach().float().clone()
        n_params = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)                                                      # vision tower and everything else frozen; PEFT unfreezes only the adapters
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", target_modules=LORA_TARGETS)
    pm = get_peft_model(model, cfg)
    inner = pm.get_base_model()
    if a.grad_ckpt:
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inner.enable_input_require_grads()
    lora_params = [(n, p) for n, p in pm.named_parameters() if p.requires_grad]
    assert lora_params and all("lora_" in n for n, _ in lora_params), "only LoRA adapters may be trainable"
    assert all("visual" not in n and "vision_tower" not in n for n, _ in lora_params), "the vision tower must stay frozen"
    n_lora = sum(p.numel() for _, p in lora_params)
    W = torch.nn.Parameter(W0.to(a.device))
    b = torch.nn.Parameter(torch.zeros(MAX_SPACE, device=a.device))
    print(f"model {a.model}@{a.revision[:8]}: {n_params / 1e9:.3f} B parameters, LoRA {n_lora / 1e6:.2f} M trainable ({len(lora_params)} tensors) + head {W.numel() / 1e3:.1f} k; mode {a.mode}; dtype {a.dtype}; noncausal={use_noncausal}", flush=True)
    fl, verdict = V.fast_path_report()
    print(f"versions: torch {torch.__version__}, transformers {__import__('transformers').__version__}, peft {__import__('peft').__version__}, " + ", ".join(fl), flush=True)
    print(f"linear-attention path: {verdict}", flush=True)
    print(f"{V.gpu_text()}; micro-batch {a.batch} x accum {a.accum} = effective {a.batch * a.accum}, epochs {a.epochs}, lr lora {a.lr_lora} head {a.lr_head}, wd {a.wd}, warmup {a.warmup}, max_len {a.max_len}, max_pixels {a.max_pixels}", flush=True)

    store = None
    if use_image:
        from utils.coco_images import ImageStore
        store = ImageStore(a.image_sha, roots=a.image_roots, cache_dir=a.image_cache, allow_download=a.download_images)

    def items_of(split):
        it = V.load_items(a.s1_data, split, a.limit)
        return it

    train = items_of("train")
    dev_full = items_of("dev")
    dev_sub = V.train_subset(dev_full, a.eval_items, a.seed) if a.eval_items else dev_full
    dev_units = V.make_units(dev_sub, "dev")
    if use_image:
        need = sorted({it["coco_id"] for it in train + dev_full + items_of("calib") + items_of("test")})
        found = sum(store.path(c) is not None for c in need)
        print(f"image store: needed {len(need)}, found locally {found}, missing {len(need) - found}", flush=True)

    meta = {"model": a.model, "revision": a.revision, "dtype": a.dtype, "params": V.checkpoint_params(a.model, a.revision) or n_params, "params_loaded": n_params, "trainable_lora_params": n_lora, "hidden_size": int(W0.shape[1]), "letter_ids": letter_ids,
            "max_pixels": a.max_pixels if use_image else 0, "min_pixels": a.min_pixels, "licence": a.licence, "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__,
            "method": "lora", "mode": a.mode, "lora": {"r": a.rank, "alpha": a.alpha, "dropout": 0.0, "targets": LORA_TARGETS, "vision_tower_frozen": True},
            "train": {"lr_lora": a.lr_lora, "lr_head": a.lr_head, "wd": a.wd, "warmup": a.warmup, "batch": a.batch, "accum": a.accum, "epochs": a.epochs, "seed": a.seed, "max_len": a.max_len,
                      "permute": "all item types" if a.permute_all else "choice items only"},
            "prompt": "shared Vietnamese letter prompt (scripts/s1_vlm_eval.py PROMPT)" + ("" if use_image else "; NO IMAGE in the input (text-only control, prompt text unchanged)"),
            "note": "training and evaluation in " + a.dtype + (" (d3 readout.safetensors[:16] + noncausal_full_attention)" if is_d3 else " (the zero-shot / head-only rows are fp16)"), "seconds_per_unit": None}
    atomic_json(os.path.join(out, "meta.json"), meta)

    def collate_train(rows):
        inp = make_inputs(proc, [r[0] for r in rows], [r[1] for r in rows], use_image)
        keep = (inp["attention_mask"].sum(1) <= a.max_len)
        if not bool(keep.all()):
            print(f"  dropped {int((~keep).sum())} over-long sample(s) (> {a.max_len} tokens) from a micro-batch", flush=True)
            if not bool(keep.any()):
                return None
            rows = [r for r, k in zip(rows, keep.tolist()) if k]
            inp = make_inputs(proc, [r[0] for r in rows], [r[1] for r in rows], use_image)
        return inp, torch.from_numpy(np.stack([r[2] for r in rows])), torch.from_numpy(np.stack([r[3] for r in rows]))

    def collate_eval(rows):
        return [r[0] for r in rows], [r[1] for r in rows], make_inputs(proc, [r[2] for r in rows], [r[3] for r in rows], use_image)

    backbone = getattr(inner, "model", inner)
    if use_noncausal:
        from transformers.masking_utils import create_recurrent_attention_mask

        def encode_hidden(inp):
            ids, present = inp["input_ids"], inp["attention_mask"]
            embeds = backbone.get_input_embeddings()(ids)
            if use_image and "pixel_values" in inp:
                pooled = backbone.get_image_features(inp["pixel_values"], inp["image_grid_thw"], return_dict=True).pooler_output
                if isinstance(pooled, (tuple, list)):
                    pooled = torch.cat(pooled, dim=0)
                pooled = pooled.to(embeds.device, embeds.dtype)
                slots, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=pooled)
                embeds = embeds.masked_scatter(slots, pooled)
                mrope = backbone.compute_3d_position_ids(
                    input_ids=ids,
                    image_grid_thw=inp["image_grid_thw"],
                    video_grid_thw=None,
                    inputs_embeds=embeds,
                    attention_mask=present,
                    past_key_values=None,
                    mm_token_type_ids=inp.get("mm_token_type_ids"),
                )
                text_positions = None
            else:
                rows, width = present.shape
                ramp = torch.arange(width, device=embeds.device).view(1, 1, -1).expand(4, rows, -1)
                text_positions, mrope = ramp[0], ramp[1:]
            lm = backbone.language_model
            rope = lm.rotary_emb(embeds, mrope)
            keys = present.bool()[:, None, None, :]
            scan = create_recurrent_attention_mask(
                config=lm.config,
                inputs_embeds=embeds,
                attention_mask=present,
                past_key_values=None,
                position_ids=text_positions,
            )
            hidden = embeds
            depth = lm.config.num_hidden_layers
            for layer, kind in zip(lm.layers[:depth], lm.config.layer_types[:depth]):
                hidden = layer(
                    hidden,
                    position_embeddings=rope,
                    attention_mask=keys if kind == "full_attention" else scan,
                    position_ids=text_positions,
                    past_key_values=None,
                    use_cache=False,
                )
            return lm.norm(hidden)[:, -1]
    else:
        def encode_hidden(inp):
            return backbone(**inp).last_hidden_state[:, -1]

    def letter_logits(inp):
        inp = {k: v.to(a.device) for k, v in inp.items()}
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32 and a.device == "cuda")):
            h = encode_hidden(inp)
        return h.float() @ W.T + b

    mon = V.Monitor(period=60.0, stall=a.watchdog_sec)

    @torch.no_grad()
    def predict(items, units):
        pm.eval()
        dl = torch.utils.data.DataLoader(EvalSet(items, units, store, use_image), batch_size=a.eval_batch, shuffle=False, num_workers=a.workers, collate_fn=collate_eval, worker_init_fn=V.worker_init)
        res = {}
        for ids, orders, inp in dl:
            ll = letter_logits(inp).cpu().numpy()
            for k in range(len(ids)):
                res[(int(ids[k]), int(orders[k]))] = ll[k]
            mon.touch()
        pm.train()
        return res

    n_train = len(train)
    per_step = a.batch * a.accum
    total_steps = max(1, math.ceil(a.epochs * n_train / per_step))
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    eval_every = max(1, total_steps // a.eval_points)
    opt = torch.optim.AdamW([{"params": [p for _, p in lora_params], "lr": a.lr_lora, "weight_decay": a.wd},
                             {"params": [W], "lr": a.lr_head, "weight_decay": a.wd},
                             {"params": [b], "lr": a.lr_head, "weight_decay": 0.0}])

    def lr_scale(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - a.warmup) / max(1, total_steps - a.warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    def trainable_state():
        return {"lora": {n: p.detach().cpu().clone() for n, p in lora_params}, "W": W.detach().cpu().clone(), "b": b.detach().cpu().clone()}

    history, best = [], {"nll": float("inf"), "step": -1, "state": None}
    t_train0 = time.time()

    def do_eval(step, epoch, train_loss):
        t = time.time()
        nll = dev_nll(dev_sub, dev_units, predict(dev_sub, dev_units))
        history.append({"step": step, "epoch": round(epoch, 3), "train_loss": train_loss, "dev_nll": nll, "dev_items": len(dev_sub), "eval_seconds": round(time.time() - t), "elapsed_s": round(time.time() - t_train0)})
        mark = ""
        if nll < best["nll"]:
            best.update(nll=nll, step=step, state=trainable_state())
            atomic_torch_save(best["state"], os.path.join(out, "best_state.pt"))
            mark = "  <- best"
        print(f"[{time.strftime('%H:%M:%S')}] EVAL step {step}/{total_steps} epoch {epoch:.2f}: dev NLL {nll:.4f} (train loss {train_loss}){mark}", flush=True)
        atomic_json(os.path.join(run, "train_log.json"), {"history": history, "best_step": best["step"], "best_dev_nll": best["nll"], "total_steps": total_steps, "config": vars(a)}, default=str)

    st = {"step": 0, "micro": 0, "loss_acc": 0.0, "loss_n": 0, "seen": 0, "run_loss": None, "epoch_idx": 0}
    order = np.random.RandomState(a.seed).permutation(n_train).tolist()
    resumed = False
    state_path = os.path.join(out, "resume", "state.pt")
    if a.resume and os.path.isfile(state_path):
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        if ck.get("signature") != resume_signature(a):
            print("resume: the saved state was made with different settings -> starting from scratch", flush=True)
        else:
            with torch.no_grad():
                for n, p in lora_params:
                    p.copy_(ck["lora"][n].to(p.device, p.dtype))
                W.copy_(ck["W"].to(a.device)); b.copy_(ck["b"].to(a.device))
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            st.update(step=ck["step"], micro=ck["step"] * a.accum, seen=ck["seen"], epoch_idx=ck["epoch_idx"])
            history[:] = ck["history"]
            best.update(nll=ck["best"]["nll"], step=ck["best"]["step"], state=ck["best"]["state"])
            order = np.random.RandomState(a.seed + st["epoch_idx"]).permutation(n_train).tolist() if st["epoch_idx"] else order
            if ck.get("rng"):
                torch.set_rng_state(ck["rng"]["torch"])
                if ck["rng"].get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
                np.random.set_state(ck["rng"]["numpy"])
            resumed = True
            print(f"RESUMED from step {st['step']}/{total_steps} ({st['seen']} items seen, best dev NLL {best['nll']:.4f} at step {best['step']})", flush=True)

    def checkpoint():
        save_resume(out, {"signature": resume_signature(a), "step": st["step"], "seen": st["seen"], "epoch_idx": st["epoch_idx"], "lora": {n: p.detach().cpu() for n, p in lora_params},
                          "W": W.detach().cpu(), "b": b.detach().cpu(), "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history,
                          "best": {"nll": best["nll"], "step": best["step"], "state": best["state"]},
                          "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "numpy": np.random.get_state()}})
        print(f"[{time.strftime('%H:%M:%S')}] checkpoint saved at step {st['step']} (resume/state.pt)", flush=True)
        if a.stop_after_step and st["step"] >= a.stop_after_step:
            print("stop-after-step reached: exiting (test)", flush=True)
            os._exit(0)

    mon.total_todo, mon.split, mon.n_shards = total_steps * per_step, "train", total_steps
    mon.start()
    print(f"{n_train} train items, {total_steps} optimizer steps, dev eval every {eval_every} steps on {len(dev_sub)} dev items ({len(dev_units)} units)", flush=True)
    if not resumed:
        if a.save_zeroshot and not os.path.isfile(os.path.join(out, "zeroshot", "views.json")):
            zs_run = os.path.join(out, "zeroshot")
            os.makedirs(zs_run, exist_ok=True)
            zs_arrays = {}
            zs_ll, zs_items = {}, {}
            for sp in ("calib", "dev", "test"):
                items = items_of(sp)
                units = V.make_units(items, sp)
                ll = predict(items, units)
                zs_ll[sp], zs_items[sp] = ll, items
                zs_arrays[sp] = V.split_arrays(items, lambda it, ll=ll: V.to_label_logits(it, {o: ll[(it["id"], o)] for o in ((1, 2) if it["questions"]["q"]["type"] == "choice" else (1,))}))
                print(f"[{time.strftime('%H:%M:%S')}] zeroshot predicted {sp}: {len(units)} units", flush=True)
            if a.zeroshot_debias:
                for sp, ll in zs_ll.items():
                    ks = sorted(ll)
                    np.savez(os.path.join(out, f"zeroshot_letters_{sp}.npz"), ids=np.array([k[0] for k in ks]), orders=np.array([k[1] for k in ks]), logits=np.stack([ll[k] for k in ks]))
                bias = V.fit_letter_position_bias({k: (None, v) for k, v in zs_ll["calib"].items()}, zs_items["calib"])
                np.save(os.path.join(out, "zeroshot_letter_bias.npy"), bias)
                deb_arrays = {sp: V.split_arrays(zs_items[sp], lambda it, ll=zs_ll[sp]: V.to_label_logits(it, {o: ll[(it["id"], o)] for o in ((1, 2) if it["questions"]["q"]["type"] == "choice" else (1,))}, letter_bias=bias)) for sp in zs_ll}
                deb_run = os.path.join(out, "zeroshot_debiased")
                os.makedirs(deb_run, exist_ok=True)
                deb_res = V.finish_run(deb_run, deb_arrays, a.s1_data, a.root)
                print(f"[{time.strftime('%H:%M:%S')}] DEBIASED zero-shot ({a.mode}) test acc (per-type T) {deb_res['test']['pertype']['all']['primary']['all']['accuracy']:.4f}", flush=True)
            zs_res = V.finish_run(zs_run, zs_arrays, a.s1_data, a.root)
            zs_acc = zs_res["test"]["pertype"]["all"]["primary"]["all"]["accuracy"]
            print(f"[{time.strftime('%H:%M:%S')}] zero-shot ({a.mode}) test acc (per-type T) {zs_acc:.4f}", flush=True)
            for vw in ("all", "core", "image_unseen"):
                p = zs_res["test"]["pertype"][vw]["primary"]
                print(f"[{time.strftime('%H:%M:%S')}] ZEROSHOT TEST {vw}: acc={p['all']['accuracy']:.4f} nll={p['all']['nll']:.4f} ece={p['all']['ece']:.4f} | noul={p['noul']['accuracy']:.4f} score={p['score']['accuracy']:.4f} choice={p['choice']['accuracy']:.4f}", flush=True)
        if a.zeroshot_only:
            print("zeroshot-only: done", flush=True)
            os._exit(0)
        do_eval(0, 0.0, None)                                                         # step 0 = zero-shot-initialised head with untrained (zero-delta) LoRA
    pm.train()

    def optimizer_step():
        torch.nn.utils.clip_grad_norm_([p for _, p in lora_params] + [W, b], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        st["step"] += 1
        step = st["step"]
        st["run_loss"] = st["loss_acc"] / max(st["loss_n"], 1)
        mon.extra = f" | step {step}/{total_steps} loss {st['run_loss']:.4f} lr {sched.get_last_lr()[0]:.2e}"
        if step % 10 == 0 or step == 1:
            print(f"[{time.strftime('%H:%M:%S')}] step {step}/{total_steps} loss(last {st['loss_n']} items) {st['run_loss']:.4f} lr_lora {sched.get_last_lr()[0]:.2e}", flush=True)
            st["loss_acc"], st["loss_n"] = 0.0, 0
        if step % eval_every == 0 or step == total_steps:
            do_eval(step, st["seen"] / n_train, round(st["run_loss"], 5))
            checkpoint()

    epoch_idx = st["epoch_idx"]
    first = resumed                                                                  # the first resumed epoch skips the items already seen in that epoch
    while st["step"] < total_steps:
        st["epoch_idx"] = epoch_idx
        epoch_seen0 = st["seen"] - epoch_idx * n_train if first else 0
        ds = TrainSet(train, store, a.seed + 7919 * epoch_idx, a.permute_all, use_image)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=order[max(epoch_seen0, 0):], num_workers=a.workers, worker_init_fn=V.worker_init, collate_fn=collate_train, drop_last=False, prefetch_factor=2 if a.workers else None)
        for batch in dl:
            if batch is None:
                continue
            inp, T, M = batch
            T, M = T.to(a.device), M.to(a.device)
            lg = letter_logits(inp).masked_fill(~M, -1e9)
            loss = -(T * torch.log_softmax(lg, -1)).sum(-1).mean()
            (loss / a.accum).backward()
            st["loss_acc"] += float(loss.detach()) * len(T)
            st["loss_n"] += len(T)
            st["micro"] += 1
            st["seen"] += len(T)
            mon.advance(len(T))
            if st["micro"] % a.accum == 0:
                optimizer_step()
                if st["step"] >= total_steps:
                    break
        else:
            if st["micro"] % a.accum and st["step"] < total_steps:                   # the epoch ended inside an accumulation group: one last step with what we have
                st["micro"] = 0
                optimizer_step()
        epoch_idx += 1
        st["epoch_idx"] = epoch_idx
        first = False
        order = np.random.RandomState(a.seed + epoch_idx).permutation(n_train).tolist()
    step = st["step"]
    train_seconds = round(time.time() - t_train0)
    print(f"training finished: {step} steps in {train_seconds}s; best dev NLL {best['nll']:.4f} at step {best['step']}", flush=True)

    # ---- restore the best state, keep the adapter + head, predict calib / dev / test
    bs = best["state"]
    with torch.no_grad():
        for n, p in lora_params:
            p.copy_(bs["lora"][n].to(p.device, p.dtype))
        W.copy_(bs["W"].to(a.device)); b.copy_(bs["b"].to(a.device))
    atomic_dir(os.path.join(out, "adapter"), lambda d: pm.save_pretrained(d))
    atomic_torch_save({"W": W.detach().cpu(), "b": b.detach().cpu(), "letter_ids": letter_ids, "best_step": best["step"]}, os.path.join(out, "head.pt"))
    arrays = {}
    t_pred = time.time()
    mon.split = "predict"
    for sp in ("calib", "dev", "test"):
        items = items_of(sp)
        units = V.make_units(items, sp)
        mon.split_total, mon.split_done = len(units), 0
        ll = predict(items, units)
        arrays[sp] = V.split_arrays(items, lambda it, ll=ll: V.to_label_logits(it, {o: ll[(it["id"], o)] for o in ((1, 2) if it["questions"]["q"]["type"] == "choice" else (1,))}))
        print(f"[{time.strftime('%H:%M:%S')}] predicted {sp}: {len(units)} units", flush=True)
    meta["seconds_per_unit"] = round((time.time() - t_pred) / max(sum(len(V.make_units(items_of(sp), sp)) for sp in ("calib", "dev", "test")), 1), 4)
    meta["train_seconds"] = train_seconds
    atomic_json(os.path.join(out, "meta.json"), meta)
    mon.stop.set()
    res = V.finish_run(run, arrays, a.s1_data, a.root)
    acc = res["test"]["pertype"]["all"]["primary"]["all"]["accuracy"]
    print(f"LoRA ({a.mode}) test acc (per-type T) {acc:.4f}; best dev NLL {best['nll']:.4f} (step {best['step']}); total {time.time() - t_begin:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
