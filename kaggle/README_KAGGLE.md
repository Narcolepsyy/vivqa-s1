# Running the ViSO experiment matrix on Kaggle

You do the clicking; nothing in this repository calls Kaggle, uses the network, or needs a key. Everything runs in a
notebook with **two T4 GPUs**: one runner process per GPU shares a job queue, resumes after interruptions and stops cleanly
before the 9-hour session limit. Training is fp32 (no mixed precision on purpose).

## What you upload (built by `python scripts/kaggle_prepare.py`, folder `~/data/kaggle_upload/`)
| folder | size | content |
|---|---|---|
| `vivqa-packed/` | ~5.0 GB | the VinVL region features of all 34,893 images as one fp16 file (+ index, tags, meta, verify.json) |
| `vivqa-s1/` | ~40 MB | the frozen ViVQA-S1 benchmark (`data/*.jsonl`, `s1bench/`, `evaluate.py`) |
| `viso-code/` | ~250 MB | `ViVQAv2-main/` with the ViSO code, configs, `kaggle/experiments.json`, the runner scripts |
Each folder has `dataset-metadata.json` (edit `"id": "<KAGGLE_USERNAME>/<slug>"` and put your Kaggle user name in; leave the slug)
and `SHA256SUMS.txt` (check with `cd <folder> && sha256sum -c SHA256SUMS.txt`).

## Steps
1. **Account.** Create / open your Kaggle account and complete phone verification (needed for GPUs and for Internet access in notebooks).
2. **Datasets.** Create the three datasets from the three folders, either on the web (Datasets -> New Dataset -> upload the folder)
   or with `kaggle datasets create -p <folder>` on your machine. **Keep them private.** Names (slugs): `vivqa-packed`, `vivqa-s1`, `viso-code`.
   Check in the dataset page that the files are all there (the packed file is 5.29 GB).
3. **Notebook.** New Notebook -> Settings: Accelerator **GPU T4 x2** (a P100 works too, then use one runner), **Internet ON**, Persistence:
   "Files only" is enough. Add the three datasets as inputs (Add Input). Run `!ls /kaggle/input/*` first: the cells below assume
   `/kaggle/input/<slug>/...`; if Kaggle nests them differently (for example an extra folder level) adjust only the paths in the first cell.
4. **Cells** (copy exactly; one cell each):

   **(a) setup**
   ```python
   !pip -q install yacs termcolor tabulate pytorch_transformers
   import os, shutil, torch
   shutil.copytree("/kaggle/input/viso-code/ViVQAv2-main", "/kaggle/working/code/ViVQAv2-main", dirs_exist_ok=True)
   os.environ["VIVQA_S1_ROOT"] = "/kaggle/input/vivqa-s1"
   print(torch.__version__, torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
   !cd /kaggle/working/code/ViVQAv2-main && PYTHONDONTWRITEBYTECODE=1 python -c "import models, tasks, data_utils.vocabs, data_utils.datasets, models.viso; print('imports ok')"
   ```
   (`transformers`, `numpy`, `pillow`, `requests`, `pyyaml` already exist in the Kaggle image. If the import line fails, copy the error to the planner.)

   **(b) preflight: run this FIRST and send `preflight.json` to the planner before the full run**
   ```python
   !cd /kaggle/working/code/ViVQAv2-main && python scripts/viso_kaggle_run.py --experiments kaggle/experiments.json --out /kaggle/working/viso_out --code-root . --s1-dir /kaggle/input/vivqa-s1 --packed /kaggle/input/vivqa-packed --device cuda:0 --preflight
   !cat /kaggle/working/viso_out/preflight.json
   ```
   It trains 200 steps of A2 and A0 on the real data and prints seconds/step, peak GPU memory, the data-loader wait share and the projected
   total GPU hours of the whole matrix (worst case = every run uses all 25 epochs; typical = 60% of that). Expected: 1-2 minutes per epoch.
   If the projected epoch time is above 3 minutes, stop and send the file to the planner (do not start the full run).

   **(c) full run: two background runners, one per GPU, deadline 8.5 h**
   ```python
   import subprocess, shutil
   shutil.rmtree("/kaggle/working/viso_out/.locks", ignore_errors=True)       # start of a session: clear stale locks
   cwd = "/kaggle/working/code/ViVQAv2-main"
   base = ["python", "scripts/viso_kaggle_run.py", "--experiments", "kaggle/experiments.json", "--out", "/kaggle/working/viso_out",
           "--code-root", ".", "--s1-dir", "/kaggle/input/vivqa-s1", "--packed", "/kaggle/input/vivqa-packed", "--deadline-hours", "8.5"]
   procs = [subprocess.Popen(base + ["--gpu", str(g)], cwd=cwd) for g in (0, 1)]
   print("started", [p.pid for p in procs])
   ```
   Watch progress in another cell: `!tail -n 4 /kaggle/working/viso_out/runner_*.log` and `!tail -n 3 /kaggle/working/viso_out/runs/*/train_log.jsonl`.
   To wait for the end in the same cell use `print([p.wait() for p in procs])` (0 = all done, 3 = stopped by the deadline, resumable, 1 = some run failed).

   **(d) finalize and download**
   ```python
   !cd /kaggle/working/code/ViVQAv2-main && python scripts/viso_kaggle_run.py --experiments kaggle/experiments.json --out /kaggle/working/viso_out --code-root . --s1-dir /kaggle/input/vivqa-s1 --packed /kaggle/input/vivqa-packed --finalize
   !ls -la /kaggle/working/viso_out/results_bundle.zip
   ```
   Download `results_bundle.zip` from the notebook's Output panel (it has the results, logs and logits but no checkpoints, < 500 MB) and send it to the planner.
   The checkpoints (`runs/*/best_model.pth`, about 160 MB each) stay in `/kaggle/working/viso_out` and are kept by "Save Version".
5. **If the session stops early** (deadline, quota, crash): click **Save Version** (Save & Run All / quick save) so that `/kaggle/working/viso_out` is stored.
   In a new session add that notebook version's output as an input and start cell (c) with one more argument:
   `"--prior-output", "/kaggle/input/<your-notebook-slug>/viso_out"` (finished runs and half-finished ones with `last.pth` are copied and continued;
   nothing is retrained). Always send `preflight.json`/`results_bundle.zip` of the finished part to the planner.
6. **Budget.** The free quota is 30 GPU-hours per week of wall-clock time; with two T4s in parallel one wall-clock hour counts once. Fill from `preflight.json`:

   | quantity | value |
   |---|---|
   | seconds per epoch ViSO / flat (preflight `projection.seconds_per_epoch`) | _fill in_ |
   | total GPU hours, worst case / typical (`projection.total_gpu_hours_*`) | _fill in_ |
   | wall-clock hours with two GPUs, worst case (`projection.wall_hours_with_two_gpus_worst_case`) | _fill in_ |
   | weekly quota | 30 h |
   If the worst-case wall-clock hours exceed what is left of the quota, run the seed-0 jobs first (they are first in the queue) and add
   `"--only", "A2_s0", "A0_s0", "A3_s0", "A3c_s0", "A1_s0", "A5_s0", "A6_s0"` to the runner arguments; run the other seeds in a later week with `--prior-output`.

## Facts the notebook relies on (re-check them in the notebook; Kaggle changes things)
- T4/P100 GPUs have 16 GB each; a session lasts at most 9 hours; `/kaggle/working` is limited to 20 GB (checkpoints of all 15 runs are about 2.4 GB);
  `/kaggle/input` is read-only. The weekly GPU quota is about 30 hours.
- The runner never touches the network. `pip install` in cell (a) needs Internet ON.
- GPU runs are not bit-exact; the CPU tests are the determinism reference. Seeds are fixed (0, 1, 2).
