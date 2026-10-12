"""External VLM baselines on ViVQA-S1 in SYSTEM ONE mode: ONE forward pass per item, a Vietnamese prompt with LETTERED options, the logits of the option-letter tokens at the answer position -> a distribution
over the item's label space (noul A/B = không / có; counts A-K = 0 ... 10 trở lên; choice A-P = the candidates). The same prompt for every model (each model's own typed-decision interface is NOT used: record that).
Stages (python scripts/s1_vlm_eval.py <stage> --model-dir <out>/<name> ...):
  features   frozen model, one pass per (item, option order): the HIDDEN STATE at the answer position (fp16) + the 16 letter logits (= zero-shot). Resumable shards. train: order 1 only; calib / dev / test:
             choice items in 2 orders (identity, reversed) averaged, noul / score once.
  zeroshot   letter logits -> label-space logits (choice: log-softmax averaged over the 2 orders) -> logits_{calib,dev,test}.npz -> per-type temperature on CALIB (viso_calibrate) -> views
  head       HEAD-ONLY adaptation: a linear head over the 16 option letters, initialised from the unembedding rows of the letter tokens (so it equals zero-shot at step 0), soft-CE on our soft labels
             (train, order 1), dev-NLL model selection, per-type temperature on calib, 3 seeds -> same outputs
  report     markdown table over every model dir under --root: zero-shot and head-only (mean +- sd of the seeds), Image-unseen first, then All / Core, per type, seen-vs-unseen gap
Frozen: the whole VLM (vision tower included). Nothing is trained except the 16 x d head. No remote code is executed (standard transformers architectures only)."""
import argparse
import faulthandler
import glob
import json
import os
import sys
import threading
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("VIVQA_S1_ROOT", os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from utils import viso_calibration as C  # noqa: E402
from utils.atomic import atomic_json, atomic_save_npy, atomic_savez, atomic_text  # noqa: E402
from utils.viso_s1 import MAX_SPACE, TYPE_IDS, label_space, read_jsonl  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOP"
TYPES = ("noul", "score", "choice")
PROMPT = ("Bạn được cho một bức ảnh và một câu hỏi trắc nghiệm. Hãy chọn đáp án đúng nhất.\nCâu hỏi: {q}\n{options}\nChỉ trả lời bằng một chữ cái.")
SPLITS = ("train", "calib", "dev", "test")


# ------------------------------------------------------------------------------------------------------------------- prompts
def option_texts(q):
    """texts of the options in label-space order (label_space(q))"""
    if q["type"] == "noul":
        return ["không", "có"]                                   # label_space = [false, true]
    if q["type"] == "score":
        return [str(i) for i in range(10)] + ["10 trở lên"]       # levels 0 .. 10 (10 = 10+)
    return list(label_space(q))


def order_perm(n, order):
    """letters -> label index: order 1 = the item's own order, order 2 = reversed"""
    return list(range(n)) if order == 1 else list(range(n))[::-1]


def build_prompt(it, order):
    q = it["questions"]["q"]
    texts = option_texts(q)
    perm = order_perm(len(texts), order)
    return PROMPT.format(q=q["instructions"], options="\n".join(f"{LETTERS[k]}. {texts[j]}" for k, j in enumerate(perm)))


def make_units(items, split):
    """[(item position, order)]: order 1 for every item, order 2 for choice items of calib / dev / test"""
    u = [(i, 1) for i in range(len(items))]
    if split != "train":
        u += [(i, 2) for i, it in enumerate(items) if it["questions"]["q"]["type"] == "choice"]
    return u


# ------------------------------------------------------------------------------------------------------------------- stage: features
def load_items(s1_data, split, limit=0):
    items = read_jsonl(os.path.join(s1_data, f"{split}.jsonl"))
    return items[:limit] if limit else items


def train_subset(items, n, seed=0):
    """seeded subset of n train items, stratified by type (proportional, largest remainder); n <= 0 or n >= len -> all items (original order kept)"""
    if not n or n >= len(items):
        return items
    by = {}
    for i, it in enumerate(items):
        by.setdefault(it["questions"]["q"]["type"], []).append(i)
    quota = {t: n * len(ix) / len(items) for t, ix in by.items()}
    take = {t: int(q) for t, q in quota.items()}
    for t in sorted(quota, key=lambda t: quota[t] - take[t], reverse=True)[:n - sum(take.values())]:
        take[t] += 1
    rng = np.random.RandomState(seed)
    keep = sorted(i for t, ix in by.items() for i in rng.permutation(ix)[:take[t]])
    return [items[i] for i in keep]


def split_tag(split, head_train_limit):
    """shard file prefix: a subset of train gets its own prefix so it can never be mixed with full-train shards"""
    return f"trainsub{head_train_limit}" if (split == "train" and head_train_limit) else split


def split_items(a, split):
    items = load_items(a.s1_data, split, a.limit)
    return train_subset(items, a.head_train_limit, a.subset_seed) if split == "train" else items


def checkpoint_params(repo, revision):
    """number of parameters STORED in the checkpoint's safetensors files (their headers), not the count of the instantiated model: transformers ties lm_head to the embeddings when the config says
    tie_word_embeddings, so a checkpoint that still ships a separate output head (OneJev-0.8B: 1.107 B) is instantiated with fewer (0.853 B). Local cache first, then the Hub metadata; None if neither works."""
    import struct
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo, revision=revision, local_files_only=True, allow_patterns=["*.safetensors"])
        total, files = 0, glob.glob(os.path.join(d, "*.safetensors"))
        for p in files:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            total += sum(int(np.prod(v["shape"])) for k, v in hdr.items() if k != "__metadata__")
        if files and total:
            return total
    except Exception:
        pass
    try:
        from huggingface_hub import get_safetensors_metadata
        return int(sum(get_safetensors_metadata(repo, revision=revision).parameter_count.values()))
    except Exception:
        return None


def fla_backend():
    """flash-linear-attention's own view of the device. Without gcc + Python headers Triton cannot build its launcher and fla SILENTLY rolls back to CPU ('cpu'): hours slower."""
    try:
        from fla.utils._device import get_available_device
        return str(get_available_device())
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def require_fast_fla(device, needed=True):
    """hard check at startup: on a cuda run fla must report the 'cuda' backend (VLM_ALLOW_SLOW_FLA=1 overrides, e.g. for a model without linear attention)"""
    be = fla_backend()
    print(f"fla backend: {be}" + ("" if needed else " (not used by this model: no linear-attention layers)"), flush=True)
    if not needed:
        return be
    if str(device).startswith("cuda") and be.startswith("unavailable"):
        print("WARNING: flash-linear-attention is not importable: the linear-attention layers use the slow torch reference", flush=True)
    elif str(device).startswith("cuda") and be != "cuda" and os.environ.get("VLM_ALLOW_SLOW_FLA") != "1":
        raise SystemExit(f"STOP: fla backend is '{be}', not 'cuda' (no C compiler / Python.h for Triton? try: apt-get install -y build-essential python3-dev). This would run hours slower; set VLM_ALLOW_SLOW_FLA=1 to override.")
    return be


def fast_path_report():
    """which implementation the linear-attention layers will actually run (transformers binds fla's Triton kernel at import if fla imports, else a slow torch reference)"""
    lines = []
    for mod in ("fla", "triton", "causal_conv1d", "flash_attn"):
        try:
            lines.append(f"{mod} {__import__(mod).__version__}")
        except Exception as e:
            lines.append(f"{mod} not importable ({type(e).__name__})")
    verdict = "unknown"
    try:
        import importlib
        M = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        res = []
        for n in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
            fn = getattr(M, n)
            cv = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))
            impl = cv.get("implementation")
            res.append(f"{n}: {'FAST ' + impl.__module__ if cv.get('is_new_implementation') else 'TORCH FALLBACK (slow reference)'}")
        verdict = "; ".join(res)
    except Exception as e:
        verdict = f"could not inspect ({type(e).__name__}: {e})"
    return lines, verdict


def gpu_text():
    if not torch.cuda.is_available():
        return "no gpu"
    free, total = torch.cuda.mem_get_info()
    return f"gpu {torch.cuda.get_device_name(0)}: alloc {torch.cuda.memory_allocated() / 2**30:.1f} GB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, used {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB"


class Monitor:
    """background thread: a heartbeat line every `period` s (even while the main thread is blocked) and a watchdog (no unit finished for `stall` s ->
    all thread stacks + the exit of the stage with rc 3; shards are written atomically, so a re-run resumes)."""

    def __init__(self, period=60.0, stall=600.0):
        self.period, self.stall = period, stall
        self.split, self.split_done, self.split_total = "", 0, 0
        self.shard, self.n_shards = 0, 0
        self.total_todo, self.session_done = 0, 0
        self.t_start = time.time()
        self.t_first = None                                                  # time of the first finished batch (the warm-up is excluded from the rate)
        self.n_first = 0
        self.last_progress = time.time()
        self.stop = threading.Event()
        self.extra = ""                                                     # free text appended to the heartbeat (e.g. the training loss)
        self._prev = (time.time(), 0)

    def advance(self, n):
        now = time.time()
        if self.t_first is None:
            self.t_first, self.n_first = now, self.session_done + n
        self.split_done += n
        self.session_done += n
        self.last_progress = now

    def touch(self):
        self.last_progress = time.time()

    def rate(self):
        if self.t_first is None or self.session_done <= self.n_first:
            return 0.0
        return (self.session_done - self.n_first) / max(time.time() - self.t_first, 1e-6)

    def line(self):
        now = time.time()
        r = self.rate()
        win = max(now - self._prev[0], 1e-6)
        recent = (self.session_done - self._prev[1]) / win
        self._prev = (now, self.session_done)
        left = self.total_todo - self.session_done
        eta = f"{left / r / 3600:.2f} h" if r > 0 else "?"
        return (f"[{time.strftime('%H:%M:%S')}] HB {self.split} shard {self.shard}/{self.n_shards} | split {self.split_done}/{self.split_total} | stage {self.session_done}/{self.total_todo} units | "
                f"{r:.2f} units/s (last {win:.0f}s: {recent:.2f}) | ETA {eta} | idle {now - self.last_progress:.0f}s | {gpu_text()}{self.extra}")

    def _run(self):
        last_hb = time.time()
        while not self.stop.wait(5.0):
            now = time.time()
            if now - last_hb >= self.period:
                print(self.line(), flush=True)
                last_hb = now
            if self.stall and now - self.last_progress >= self.stall:
                print(f"[{time.strftime('%H:%M:%S')}] WATCHDOG: no unit finished for {now - self.last_progress:.0f}s (split {self.split}, shard {self.shard}); dumping all thread stacks and stopping the stage (rc 3). "
                      f"Finished shards are kept; re-run to resume.", flush=True)
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                sys.stderr.flush(); sys.stdout.flush()
                os._exit(3)

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="monitor").start()
        return self


def worker_init(_worker_id):
    """DataLoader workers do image decode + tokenisation: one math thread each (VLM_WORKER_THREADS), else N workers x all cores oversubscribe the CPU (load 110 on 32 vCPU starved the GPU)"""
    n = int(os.environ.get("VLM_WORKER_THREADS", "1"))
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(n)


def shard_ok(path, expected):
    """a shard counts as done only if it loads and holds exactly the expected number of units: shards copied back from another machine (rsync) or cut by a crash are renamed *.bad and recomputed"""
    if not os.path.isfile(path):
        return False
    try:
        z = np.load(path)
        n = len(z["ids"])
        ok = n == expected and len(z["order"]) == n and z["h"].shape[0] == n and z["ll"].shape[0] == n
    except Exception:
        ok = False
    if not ok:
        print(f"shard check: {path} is unreadable or has the wrong size (want {expected} units) -> recomputing it", flush=True)
        os.replace(path, path + ".bad")
    return ok


def stage_features(a):
    from vlm_adapters import make_adapter
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    out = a.model_dir
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    ad = make_adapter(a, dtype)                                              # standard transformers models; trust_remote_code only for the pinned Vintern revision (scripts/vlm_adapters.py)
    require_fast_fla(a.device, needed=ad.needs_linear_attention)
    tok, letter_ids, W = ad.tok, ad.letter_ids, ad.W
    atomic_save_npy(os.path.join(out, "letter_rows.npy"), W.cpu().numpy())
    n_params = ad.n_params
    meta = {"model": a.model, "revision": a.revision, "dtype": a.dtype, "params": checkpoint_params(a.model, a.revision) or n_params, "params_loaded": n_params, "hidden_size": int(W.shape[1]), "letter_ids": letter_ids,
            "max_pixels": a.max_pixels, "min_pixels": a.min_pixels,
            "prompt": "shared Vietnamese letter prompt (scripts/s1_vlm_eval.py PROMPT); the model's own typed-decision interface is NOT used", "licence": a.licence, "transformers": __import__("transformers").__version__,
            **ad.meta_extra, "head_train_limit": a.head_train_limit, "subset_seed": a.subset_seed}
    atomic_json(os.path.join(out, "meta.json"), meta)
    print(f"model {a.model}@{a.revision[:8]}: {n_params / 1e9:.3f} B parameters, hidden {W.shape[1]}, dtype {a.dtype}, device {a.device}", flush=True)
    fl, verdict = fast_path_report()
    print(f"versions: python {sys.version.split()[0]}, torch {torch.__version__}, transformers {meta['transformers']}, " + ", ".join(fl), flush=True)
    print(f"linear-attention path: {verdict}", flush=True)
    print(f"{gpu_text()}; batch {a.batch}, workers {a.workers}, shard {a.shard}, max_pixels {a.max_pixels}, min_pixels {a.min_pixels}, max_hours {a.max_hours}, watchdog {a.watchdog_sec}s"
          + (f", TRAIN SUBSET {a.head_train_limit} (seed {a.subset_seed}, stratified by type)" if a.head_train_limit else ""), flush=True)

    adapter = ad

    class Units(Dataset):
        def __init__(self, items, units, store):
            self.items, self.units, self.store = items, units, store

        def __len__(self):
            return len(self.units)

        def __getitem__(self, k):
            i, order = self.units[k]
            it = self.items[i]
            return it["id"], order, adapter.prepare_image(self.store.open(it["coco_id"])), build_prompt(it, order)

    def collate(rows):
        return rows

    from utils.coco_images import ImageStore
    store = ImageStore(a.image_sha, roots=a.image_roots, cache_dir=a.image_cache, allow_download=a.download_images)       # every image read is verified against the sha256 list
    plan = {}
    for split in a.splits:
        items = split_items(a, split)
        units = make_units(items, split)
        tag = split_tag(split, a.head_train_limit)
        todo = [s0 for s0 in range(0, len(units), a.shard) if not shard_ok(os.path.join(out, "shards", f"{tag}_{s0 // a.shard:05d}.npz"), len(units[s0:s0 + a.shard]))]
        plan[split] = (items, units, tag, todo)
    need = sorted({it["coco_id"] for sp in plan for it in plan[sp][0]})
    n_found = sum(store.path(c) is not None for c in need)
    print(f"image store: {len(store.index)} files matched under {a.image_roots}; images needed by {list(a.splits)}: {len(need)}, found locally {n_found}, missing {len(need) - n_found} "
          f"({'will be downloaded from images.cocodataset.org and sha256-verified' if a.download_images else 'download is OFF -> the run will fail on the first missing image'}); example path: {store.path(need[0])}", flush=True)
    if len(need) - n_found and a.download_images:
        miss = next(c for c in need if store.path(c) is None)
        t = time.time()
        store.read_bytes(miss)                                                # fail now, not after 10 minutes, when the internet is off
        print(f"download probe ok: coco_id {miss} fetched and verified in {time.time() - t:.1f}s", flush=True)
    for split, (items, units, tag, todo) in plan.items():
        print(f"plan {split}: {len(items)} items, {len(units)} units, shards {len(todo)}/{(len(units) + a.shard - 1) // a.shard} still to do", flush=True)
    total_units = sum(len(plan[sp][1]) for sp in plan)
    mon = Monitor(period=a.heartbeat_sec, stall=a.watchdog_sec)
    mon.total_todo = sum(len(plan[sp][1][s0:s0 + a.shard]) for sp in plan for s0 in plan[sp][3])
    if a.preflight_minutes:
        mon.total_todo = min(mon.total_todo, total_units)
    print(f"{mon.total_todo} units to do in this stage", flush=True)
    mon.start()
    t_all, n_all, t_start = 0.0, 0, time.time()
    stopped = False
    for split, (items, units, tag, todo) in plan.items():
        mon.split, mon.split_total, mon.n_shards = split, len(units), (len(units) + a.shard - 1) // a.shard
        mon.split_done = len(units) - sum(len(units[s0:s0 + a.shard]) for s0 in todo)
        for s0 in range(0, len(units), a.shard):
            if a.max_hours and time.time() - t_start > a.max_hours * 3600:
                print(f"time budget of {a.max_hours} h reached: stopping after the finished shards (re-run with the shards attached to continue)", flush=True)
                stopped = True
                break
            path = os.path.join(out, "shards", f"{tag}_{s0 // a.shard:05d}.npz")
            if os.path.isfile(path):
                continue
            mon.shard = s0 // a.shard + 1
            chunk = units[s0:s0 + a.shard]
            dl = DataLoader(Units(items, chunk, store), batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate, worker_init_fn=worker_init)
            ids, orders, hs, lls = [], [], [], []
            t0 = time.time()
            mon.last_progress = t0
            for rows in dl:
                h = ad.encode(rows, a.device)
                ids += [r[0] for r in rows]
                orders += [r[1] for r in rows]
                hs.append(h.cpu().numpy().astype(np.float16))
                lls.append((h @ W.T).cpu().numpy())
                mon.advance(len(rows))
                if a.preflight_minutes and time.time() - t_start > a.preflight_minutes * 60:
                    stopped = True
                    break
            if stopped and a.preflight_minutes:
                break                                                         # the partial shard is dropped
            atomic_savez(path, ids=np.asarray(ids), order=np.asarray(orders), h=np.concatenate(hs), ll=np.concatenate(lls))
            dt = time.time() - t0
            t_all += dt
            n_all += len(chunk)
            print(f"[{time.strftime('%H:%M:%S')}] {split} shard {s0 // a.shard}: {len(chunk)} units in {dt:.0f}s ({len(chunk) / dt:.1f} units/s) saved; {mon.session_done}/{mon.total_todo} units of this stage done", flush=True)
            meta["seconds_per_unit"] = round(t_all / max(n_all, 1), 4)
            atomic_json(os.path.join(out, "meta.json"), meta)
        if stopped:
            break
    mon.stop.set()
    if a.preflight_minutes:
        r = mon.rate()
        full = {sp: len(make_units(load_items(a.s1_data, sp), sp)) for sp in SPLITS}
        n_full = sum(full.values())
        sub = n_full - full["train"] + len(make_units(train_subset(load_items(a.s1_data, "train"), 8000, a.subset_seed), "train"))
        rep = {"model": a.model, "units_measured": mon.session_done, "units_per_s_after_warmup": round(r, 3), "seconds_per_unit": round(1 / r, 4) if r else None,
               "units_full_run": n_full, "projected_hours_full_run": round(n_full / r / 3600, 2) if r else None,
               "units_with_8000_train_subset": sub, "projected_hours_8000_train_subset": round(sub / r / 3600, 2) if r else None,
               "linear_attention_path": verdict, "gpu": gpu_text(), "images_found": n_found, "images_missing": len(need) - n_found,
               "note": "rate = units finished after the first batch / time since the first batch; includes image decode + sha256 + download of missing images"}
        atomic_json(os.path.join(out, "preflight.json"), rep)
        print("PREFLIGHT RESULT: " + json.dumps(rep), flush=True)


# ------------------------------------------------------------------------------------------------------------------- assembling
def shards_complete(model_dir, s1_data, split, limit=0, head_train_limit=0, subset_seed=0):
    """(units found in the shards, units expected) of a split (train with head_train_limit: the subset's own shards)"""
    got = 0
    for p in glob.glob(os.path.join(model_dir, "shards", f"{split_tag(split, head_train_limit)}_*.npz")):
        if not p.endswith(".tmp.npz"):
            got += len(np.load(p)["ids"])
    items = load_items(s1_data, split, limit)
    if split == "train":
        items = train_subset(items, head_train_limit, subset_seed)
    return got, len(make_units(items, split))


def load_shards(model_dir, split):
    """-> {(item id, order): (h fp16, letter logits)}"""
    d = {}
    for p in sorted(glob.glob(os.path.join(model_dir, "shards", f"{split}_*.npz"))):
        if p.endswith(".tmp.npz"):
            continue
        z = np.load(p)
        for k in range(len(z["ids"])):
            d[(int(z["ids"][k]), int(z["order"][k]))] = (z["h"][k], z["ll"][k])
    return d


def to_label_logits(item, letter_logits_by_order):
    """letter logits of 1-2 orders -> label-space log-probabilities [16] (average of the orders' log-softmax over the valid letters)"""
    n = len(label_space(item["questions"]["q"]))
    acc = np.zeros(MAX_SPACE, np.float64)
    for order, ll in letter_logits_by_order.items():
        z = ll[:n].astype(np.float64)
        lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        for k, j in enumerate(order_perm(n, order)):
            acc[j] += lp[k]
    return acc / len(letter_logits_by_order)


def split_arrays(items, logits_fn):
    """-> ids, type_id, space_mask, logits for the items with a result (logits_fn(item) -> [16] or None)"""
    ids, ty, mask, lg = [], [], [], []
    for it in items:
        z = logits_fn(it)
        if z is None:
            continue
        n = len(label_space(it["questions"]["q"]))
        m = np.zeros(MAX_SPACE, bool)
        m[:n] = True
        ids.append(it["id"]); ty.append(TYPE_IDS[it["questions"]["q"]["type"]]); mask.append(m); lg.append(np.where(m, z, 0.0))
    return np.asarray(ids), np.asarray(ty), np.asarray(mask), np.asarray(lg, np.float32)


def zero_shot_logits(model_dir, items, split):
    sh = load_shards(model_dir, split)

    def fn(it):
        got = {o: sh[(it["id"], o)][1] for o in (1, 2) if (it["id"], o) in sh}
        want = 2 if (split != "train" and it["questions"]["q"]["type"] == "choice") else 1
        return to_label_logits(it, got) if len(got) == want else None
    return split_arrays(items, fn)


# ------------------------------------------------------------------------------------------------------------------- calibration + views
def evaluate_views(run_dir, s1_data, root):
    """per-type temperatures were fitted on calib by viso_calibrate: metrics of the uncalibrated and the per-type calibrated predictions on test (All / Core / Image-unseen) and dev"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s1_evaluate", os.path.join(root, "evaluate.py"))
    EV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EV)
    temps = json.load(open(os.path.join(run_dir, "calibration", "temperatures.json")))["per_type_nll"]
    t = {int(k): v for k, v in temps.items()}
    from tasks.typed_decision_task import preds_dict
    res = {"temperatures": temps}
    flags = os.path.join(s1_data, "eval_flags.jsonl")
    for split in ("dev", "test"):
        lg = C.load_logits(os.path.join(run_dir, f"logits_{split}.npz"))
        by_id = {it["id"]: it for it in read_jsonl(os.path.join(s1_data, f"{split}.jsonl"))}
        items = [by_id[int(i)] for i in lg["ids"]]
        for name, temps_ in (("uncal", None), ("pertype", t)):
            probs = C.probs_from_logits(lg["logits"], lg["space_mask"], lg["type_id"], temps_)
            preds = preds_dict(items, probs)
            r = {"all": EV.evaluate(items, preds)}
            if split == "test" and os.path.isfile(flags):
                _, views = EV.view_blocks(items, preds, flags, f"{split}.jsonl")
                r.update(views)
            res.setdefault(split, {})[name] = r
    atomic_json(os.path.join(run_dir, "views.json"), res, ensure_ascii=False)
    return res


def finish_run(run_dir, ids_arrays, s1_data, root):
    """write logits_{calib,dev,test}.npz, calibrate on calib (viso_calibrate), evaluate"""
    for split, (ids, ty, mask, lg) in ids_arrays.items():
        C.save_logits(os.path.join(run_dir, f"logits_{split}.npz"), ids, ty, lg, mask)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import viso_calibrate as VC
    VC.main(["--run-dir", run_dir, "--s1-dir", s1_data])
    return evaluate_views(run_dir, s1_data, root)


# ------------------------------------------------------------------------------------------------------------------- stage: zeroshot / head
def stage_zeroshot(a):
    arrays = {sp: zero_shot_logits(a.model_dir, load_items(a.s1_data, sp, a.limit), sp) for sp in ("calib", "dev", "test")}
    run = os.path.join(a.model_dir, "zeroshot")
    os.makedirs(run, exist_ok=True)
    res = finish_run(run, arrays, a.s1_data, a.root)
    print("zero-shot test (per-type T) overall acc:", res["test"]["pertype"]["all"]["primary"]["all"]["accuracy"])


def head_inputs(model_dir, items, split, orders_wanted):
    sh = load_shards(model_dir, split)
    keep = [it for it in items if all((it["id"], o) in sh for o in orders_wanted(it))]
    return keep, sh


def soft_targets(items, order=1):
    """soft labels in LETTER order (order 1 = label order): [N, 16]"""
    t = np.zeros((len(items), MAX_SPACE), np.float32)
    m = np.zeros((len(items), MAX_SPACE), bool)
    for i, it in enumerate(items):
        q = it["questions"]["q"]
        sp = label_space(q)
        m[i, :len(sp)] = True
        for k, j in enumerate(order_perm(len(sp), order)):
            t[i, k] = it["gold"]["q"]["probabilities"].get(sp[j], 0.0)
    return t, m


def train_head(a, seed):
    W0 = torch.from_numpy(np.load(os.path.join(a.model_dir, "letter_rows.npy")))
    tr_items = split_items(a, "train")
    tr_items, sh_tr = head_inputs(a.model_dir, tr_items, split_tag("train", a.head_train_limit), lambda it: (1,))
    H = torch.from_numpy(np.stack([sh_tr[(it["id"], 1)][0] for it in tr_items]).astype(np.float32))
    T, M = soft_targets(tr_items)
    T, M = torch.from_numpy(T), torch.from_numpy(M)
    dev_items = load_items(a.s1_data, "dev", a.limit)
    dev_items, sh_dev = head_inputs(a.model_dir, dev_items, "dev", lambda it: (1, 2) if it["questions"]["q"]["type"] == "choice" else (1,))
    Hd = {k: torch.from_numpy(v[0].astype(np.float32)) for k, v in sh_dev.items()}
    gold = C.gold_arrays(dev_items)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(W0.clone())
    b = torch.nn.Parameter(torch.zeros(MAX_SPACE))
    opt = torch.optim.AdamW([W, b], lr=a.lr, weight_decay=a.wd)

    def dev_nll(Wc, bc):
        nll, n = 0.0, 0
        with torch.no_grad():
            for i, it in enumerate(dev_items):
                orders = (1, 2) if it["questions"]["q"]["type"] == "choice" else (1,)
                lg = {o: (Hd[(it["id"], o)] @ Wc.T + bc).numpy() for o in orders}
                z = to_label_logits(it, lg)
                nn_ = len(label_space(it["questions"]["q"]))
                p = np.exp(z[:nn_] - z[:nn_].max()); p /= p.sum()
                nll -= np.log(max(p[gold["gold_idx"][i]], 1e-12)); n += 1
        return nll / max(n, 1)

    best, best_state, hist = float("inf"), (W.detach().clone(), b.detach().clone()), []
    hist.append({"epoch": 0, "dev_nll": dev_nll(W.detach(), b.detach())})
    bad = 0
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(len(H), generator=g)
        for s in range(0, len(H), a.batch_head):
            ix = perm[s:s + a.batch_head]
            x = torch.nn.functional.dropout(H[ix], p=a.dropout, training=True)
            lg = (x @ W.T + b).masked_fill(~M[ix], -1e9)
            loss = -(T[ix] * torch.log_softmax(lg, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = dev_nll(W.detach(), b.detach())
        hist.append({"epoch": ep, "dev_nll": v})
        if v < best:
            best, bad, best_state = v, 0, (W.detach().clone(), b.detach().clone())
        else:
            bad += 1
            if bad >= a.patience:
                break
    return best_state, hist


def stage_head(a):
    results = {}
    for seed in range(a.seeds):
        (W, b), hist = train_head(a, seed)
        arrays = {}
        for sp in ("calib", "dev", "test"):
            items = load_items(a.s1_data, sp, a.limit)
            sh = load_shards(a.model_dir, sp)

            def fn(it, sh=sh, sp=sp):
                orders = (1, 2) if (sp != "train" and it["questions"]["q"]["type"] == "choice") else (1,)
                if not all((it["id"], o) in sh for o in orders):
                    return None
                return to_label_logits(it, {o: (torch.from_numpy(sh[(it["id"], o)][0].astype(np.float32)) @ W.T + b).numpy() for o in orders})
            arrays[sp] = split_arrays(items, fn)
        run = os.path.join(a.model_dir, f"head_s{seed}")
        os.makedirs(run, exist_ok=True)
        atomic_json(os.path.join(run, "head_log.json"), {"seed": seed, "head_train_limit": a.head_train_limit, "lr": a.lr, "weight_decay": a.wd, "dropout": a.dropout, "history": hist, "best_dev_nll": min(h["dev_nll"] for h in hist), "zero_shot_dev_nll": hist[0]["dev_nll"]})
        res = finish_run(run, arrays, a.s1_data, a.root)
        results[seed] = res["test"]["pertype"]["all"]["primary"]["all"]["accuracy"]
        print(f"head seed {seed}: dev NLL {hist[0]['dev_nll']:.4f} (zero-shot) -> {min(h['dev_nll'] for h in hist):.4f}; test acc (per-type T) {results[seed]:.4f}", flush=True)


# ------------------------------------------------------------------------------------------------------------------- report
def cell(runs, view, metric, typ="all", variant="pertype"):
    vals = []
    for r in runs:
        try:
            vals.append(r["test"][variant][view]["primary"][typ][metric] if view != "all" else r["test"][variant]["all"]["primary"][typ][metric])
        except KeyError:
            return "-"
    m, s = float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)
    return f"{m:.3f}" + (f" ± {s:.3f}" if len(vals) > 1 else "")


def true_params(meta, cache):
    """parameters in the checkpoint: meta['params'] of new runs already is that count; older meta files hold the instantiated count, so ask the safetensors headers / the Hub (cached per model)"""
    if "params_loaded" in meta:
        return meta["params"]
    key = (meta["model"], meta["revision"])
    if key not in cache:
        cache[key] = checkpoint_params(*key)
    return cache[key] or meta["params"]


def stage_report(a):
    L = ["# External VLM baselines in System One mode (frozen model; letter-token logits; same Vietnamese prompt)", "",
         "Test split, per-type temperature fitted on calib. Head-only = linear head over the 16 option letters initialised from the unembedding rows (3 seeds, mean ± sd).", ""]
    rows, notes, ckpt_cache = [], [], {}
    for md in sorted(glob.glob(os.path.join(a.root_models, "*", "meta.json"))):
        d = os.path.dirname(md)
        meta = json.load(open(md))
        name = os.path.basename(d)
        zs = [json.load(open(p)) for p in glob.glob(os.path.join(d, "zeroshot", "views.json"))]
        hd = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "head_s*", "views.json")))]
        lo = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "lora_s*", "views.json")))]      # scripts/s1_vlm_lora.py
        rows.append((name, meta, zs, hd, lo))
    L += ["| model | params | licence | method | Image-unseen acc | NLL | ECE | All acc | Core acc | seen - unseen acc | noul | score | choice |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, meta, zs, hd, lo in rows:
        for method, runs in (("zero-shot", zs), ("head-only", hd), ("LoRA + head", lo)):
            if not runs:
                continue
            gap = "-"
            try:
                def seen_minus_unseen(r):                                          # Core-seen accuracy (image seen in train) minus Image-unseen accuracy; Image-unseen is a subset of Core
                    c, u = r["test"]["pertype"]["core"]["primary"]["all"], r["test"]["pertype"]["image_unseen"]["primary"]["all"]
                    n_seen = c["n"] - u["n"]
                    return (c["accuracy"] * c["n"] - u["accuracy"] * u["n"]) / n_seen - u["accuracy"]
                gaps = [seen_minus_unseen(r) for r in runs]
                gap = f"{np.mean(gaps):+.3f}"
            except KeyError:
                pass
            true_p = true_params(meta, ckpt_cache)
            loaded_p = meta.get("params_loaded", meta["params"])
            flag = " †" if abs(true_p - loaded_p) / true_p > 0.05 else ""       # stored vs instantiated differ clearly (e.g. a separate output head that transformers ties away)
            if flag and name not in [n for n, _ in notes]:
                txt = (f"{name} ({meta['model']}): the checkpoint stores {true_p / 1e9:.3f} B parameters (the safetensors files)" + (", i.e. MORE than 1 B" if true_p > 1.0e9 else "") +
                       f", but transformers instantiates {loaded_p / 1e9:.3f} B: the checkpoint ships an output head separate from the input embeddings while its config says tie_word_embeddings, so the stored head is not loaded.")
                if "OneJev" in meta["model"]:
                    txt += " For OneJev-0.8B the 16 letter rows used for the zero-shot logits and as the head initialisation were checked to be identical in the head and in the embedding matrix (bf16, exact), so the results do not depend on this."
                notes.append((name, txt))
            L.append(f"| {name} | {true_p / 1e9:.2f} B{flag} | {meta.get('licence', '')} | {method} | {cell(runs, 'image_unseen', 'accuracy')} | {cell(runs, 'image_unseen', 'nll')} | {cell(runs, 'image_unseen', 'ece')} | "
                     f"{cell(runs, 'all', 'accuracy')} | {cell(runs, 'core', 'accuracy')} | {gap} | " + " | ".join(cell(runs, 'all', 'accuracy', t) for t in TYPES) + " |")
    if notes:
        L += ["", *[f"† {t}" for _, t in notes]]
    L += ["", "Per model: revision, dtype, tile cap, seconds per unit (one forward pass), prompt:", ""]
    for name, meta, zs, hd, lo in rows:
        L.append(f"- **{name}**: {meta['model']}@{meta['revision'][:8]}, {meta['dtype']}, max_pixels {meta['max_pixels']}, {meta.get('seconds_per_unit', '?')} s per unit, transformers {meta['transformers']}; {meta['prompt']}")
    atomic_text(a.report_file or os.path.join(a.root_models, "vlm_report.md"), "\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "zeroshot", "head", "report"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--licence", default="")
    ap.add_argument("--model-dir", default=None, help="<out>/<short name>")
    ap.add_argument("--root-models", default=None, help="report: the directory holding one folder per model")
    ap.add_argument("--report-file", default=None, help="report: where to write the markdown (default <root-models>/vlm_report.md)")
    ap.add_argument("--s1-data", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--root", default=os.path.join(REPO, "..", "vivqa_s1_v2"), help="benchmark root (evaluate.py)")
    ap.add_argument("--image-roots", nargs="*", default=[os.path.join(REPO, "..", "vivqav2_dataset", "images")], help="directories searched (recursively) for COCO / benchmark image files, e.g. /kaggle/input")
    ap.add_argument("--image-sha", default=os.path.join(REPO, "kaggle", "s1_images_sha256.tsv.gz"))
    ap.add_argument("--image-cache", default=None, help="where downloaded COCO files are kept")
    ap.add_argument("--download-images", action="store_true", help="fallback: download missing images from images.cocodataset.org (verified against the sha list)")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=0, help="first N items of every split (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shard", type=int, default=512)
    ap.add_argument("--max-hours", type=float, default=0.0, help="features: stop after the current shard once this many hours have passed (resumable)")
    ap.add_argument("--preflight-minutes", type=float, default=0.0, help="features: measure the speed only: stop after this many minutes (inside a shard; the partial shard is dropped) and write preflight.json")
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--watchdog-sec", type=float, default=600.0, help="features: no unit finished for this long -> dump all stacks, exit the stage with rc 3 (0 = off)")
    ap.add_argument("--head-train-limit", type=int, default=0, help="use a seeded subset of this many train items, stratified by type (features AND head; its shards are named trainsub<N>_*); 0 = all (default)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true", help="ONLY for the pinned Vintern revision (scripts/vlm_adapters.py checks the revision and the sha256 of its 5 remote files); any other model is refused")
    ap.add_argument("--max-tiles", type=int, default=4, help="Vintern: maximum number of 448 px dynamic tiles (+ a thumbnail when > 1), 256 image tokens each; recorded in meta.json")
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-head", type=int, default=256)
    a = ap.parse_args(argv)
    faulthandler.enable()
    a.s1_data, a.root = (os.path.abspath(os.path.expanduser(p)) for p in (a.s1_data, a.root))
    a.image_roots = [os.path.abspath(os.path.expanduser(p)) for p in a.image_roots]
    {"features": stage_features, "zeroshot": stage_zeroshot, "head": stage_head, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
