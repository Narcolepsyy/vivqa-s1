# Chạy VLM trên GPU thuê (EZYCLOUDX, Docker GPU, RTX 3090) / Run the VLM jobs on the rented GPU

Mọi thứ trên máy thuê **bị xóa khi hết hạn thuê**. Kết quả được đẩy lên Kaggle dataset **riêng tư** `narcolepsyy/vivqa-vlm-results` sau MỖI mô hình.
Everything on the rented machine is **wiped when the rental ends**. Results are uploaded after EACH job to the PRIVATE Kaggle dataset `narcolepsyy/vivqa-vlm-results`.

You need / Bạn cần: a **Kaggle API token** (kaggle.com → Settings → API → Create new token; make one just for this and revoke it afterwards / tạo token riêng, thu hồi sau khi xong).
The scripts never write the token to disk; they ask for it with a hidden prompt (typing shows nothing) / script chỉ hỏi token bằng ô nhập ẩn, không ghi ra đĩa.

## 1. Start the container / Khởi động container
1. EZYCLOUDX → product **Docker GPU** → RTX 3090 24 GB, a CUDA image with python3 (≥ 3.10), ≥ 150 GB disk. Open the **terminal** of the container.
   Chọn Docker GPU, RTX 3090, image có CUDA + python3, đĩa ≥ 150 GB, mở terminal.
2. Check / kiểm tra: `nvidia-smi` shows the 3090; `python3 --version` ≥ 3.10; `df -h /workspace` (or `df -h ~`) ≥ 150 GB free.

## 2. Put the scripts on the machine / Đưa script lên máy
Either (a) **paste** the whole block from `ezy_paste.txt` into the terminal (it creates `ezy/` and checks the checksum), or (b) upload `ezy_bundle.tar.gz` with the web file manager and run `tar xzf ezy_bundle.tar.gz`.
Hoặc (a) dán toàn bộ khối trong `ezy_paste.txt`, hoặc (b) tải `ezy_bundle.tar.gz` lên rồi giải nén.

## 3. Open tmux (so a dropped browser tab does not kill the run) / Mở tmux
```
cd ezy          # the folder from step 2
tmux new -s run # if tmux is missing: apt-get update && apt-get install -y tmux
```
Reattach later with / vào lại: `tmux attach -t run`. Detach / thoát mà vẫn chạy: `Ctrl-b` then `d`.

## 4. Setup (once, ~1–2 h, mostly downloads) / Cài đặt (một lần)
```
bash setup.sh
```
- Asks the Kaggle token once (hidden). Downloads viso-code + vivqa-s1 (Kaggle), the COCO zips (~19 GB, then extracts only the 34,893 benchmark images and checks every sha256; **stops on any mismatch**), and 7 Hugging Face models (~45 GB).
- Ends with two smoke tests (pipeline on 24 items per split; LoRA 3 steps). Look at the printed lines:
  - `linear-attention path: ... FAST fla...` is good (`TORCH FALLBACK` = slow, tell the planner).
  - `status: complete` for the dry run and a `LoRA (img) test acc ...` line.
- Safe to run again: finished steps are skipped. To skip steps: `EZY_SKIP="coco models" bash setup.sh`.

## 5. (Optional, 10 min) measure the speed / đo tốc độ
```
bash run.sh preflight
```
Prints units/s and the projected hours per small model for the full train and for `VLM_HEAD_TRAIN_LIMIT=8000`.
If the full train is too slow / too expensive, `export VLM_HEAD_TRAIN_LIMIT=8000` before step 6 (the planner decides). LoRA always uses the full train.

## 6. Run the jobs / Chạy
```
bash run.sh all
```
= `intern jpt onejev` (zero-shot + head-only) → `lora-img lora-txt` (LoRA fine-tune with images / text-only control) → `ref-intern2b ref-intern4b ref-qwen3vl8b` (zero-shot only).
Subsets: `bash run.sh intern jpt onejev`, `bash run.sh lora`, `bash run.sh ref`, `bash run.sh qwen` (only if the Kaggle `vlm-qwen` ended incomplete).
**No Kaggle token is needed** in the default mode. Results stay in `<root>/results/` and are pulled to the PC by **rsync** (use `--exclude='*.tmp*'`); `results/RUN_STATUS.json` lists, per job, the status, seconds and files.
Env options / tùy chọn: `VLM_HEAD_TRAIN_LIMIT=8000`, `FORCE=1` (rerun a finished job), `LORA_ARGS="--batch 4 --accum 8"` (LoRA out of GPU memory), `VLM_BATCH=2` (8B out of memory), `MODEL_CLEANUP=1` (small disk), `KAGGLE_UPLOAD=1` (optional: also upload each finished job to the private Kaggle dataset; then it asks for the token and prints SAFE TO END RENTAL).

### If the machine dies / Nếu máy bị xóa
Resume = run the same command again. Finished jobs are skipped (`state/done_<job>`), finished feature shards are reused (each is checked: unreadable or wrong-size shards are recomputed), and a LoRA job continues from its last checkpoint (`<job>/resume/`, written at every 1/4 epoch). If you copy results back from the PC into `<root>/results/`, run `bash run.sh restore` (or just `bash run.sh ...`): it re-creates the `done_<job>` markers from complete summaries.

## 7. Check progress / Xem tiến độ
- In tmux: a **heartbeat line every 60 s** (units done/expected, units/s, ETA, GPU memory). LoRA also prints `step N/total loss ...`, `EVAL ... dev NLL` and `checkpoint saved`.
- `cat results/RUN_STATUS.json`; another tmux window (`Ctrl-b c`): `nvidia-smi`, `tail -f results/*_features.log`.
- A stage that makes no progress for 10 min dumps its stack trace and stops by itself; run the same command again.

## 8. When can I end the rental? / Khi nào được tắt máy?
When the terminal prints **`RUN COMPLETE`** (every job finished) **and your rsync pull has copied `results/`**. If it prints `RUN NOT COMPLETE`, run the same command again.
Khi thấy `RUN COMPLETE` và đã rsync xong thư mục `results/` về máy. Nếu thấy `RUN NOT COMPLETE` hãy chạy lại đúng lệnh cũ.
(With `KAGGLE_UPLOAD=1` the signal is `SAFE TO END RENTAL`; revoke the token afterwards.)
- Per job in `results/<job>/`: `meta.json`, `zeroshot/`, `head_s*/` or `lora_s0/` (+ `adapter/`, `head.pt` for LoRA); plus `summary_<job>.json`, logs, `run_info`, `vlm_report.md`.

## 9. Troubleshooting
- Browser terminal disconnected: `tmux attach -t run`. If tmux was not used and the run died: run the same command again (it resumes).
- COCO download slow/interrupted: re-run `bash setup.sh` (curl resumes).
- Updated scripts from the planner: copy the new `ezy/` files over the old ones and run `bash apply_update.sh`.
- Anything unclear: copy the last 40 lines of the terminal to the planner.

## 10. ViSO-FT (live encoders, top 4 blocks fine-tuned)
```
bash run_viso_ft.sh 0          # seed 0 -> results/viso-ft-bami-s0 ; waits until the rental queue is idle
```
- Needs the images (setup.sh) and, for the paired comparison, the predictions of the reference runs: `<root>/refs/ABL_bami_s0/preds_{dev,test}.jsonl` and `<root>/refs/A2_s0/...` (copy them from the matrix output). Without them the training still runs and the comparison is skipped.
- First it builds a uint8 image cache (~6.9 GB, a few minutes, reused), then trains (12 epochs, patience 5, best-dev-NLL; `last.pth` every epoch = resumable: just run the same command again), then writes logits / predictions for calib / dev / test, the per-type temperature, and `ft_report/` (paired bootstrap vs the references + the image-shuffle check).
- It prints and stores `PARAMETERS trained ... of ...`, seconds per epoch and the peak GPU memory (`ft_run.json`, `train_log.jsonl`).
- `rsync` tip: the job dir has `last.pth` (1.2 GB, only needed to resume) and `best_model.pth` (0.4 GB): exclude `last.pth` from the pull once the job is complete.
- The LoRA variant (`FT_EXTRA="--enc-lora 16"`) exists but is NOT run by default.

## 11. Vintern-1B-v3_5 (the one `trust_remote_code` model)
```
bash run_vintern.sh            # results/vintern ; resumable ; no waiting for other jobs (queue it yourself)
```
- `5CD-AI/Vintern-1B-v3_5` (MIT, 0.938 B, InternVLChatModel) is the ONLY model run with remote code, and only at the pinned revision `b98f263e…`. The harness refuses any other revision; the sha256 of its five remote
  files (`configuration_intern_vit.py`, `configuration_internvl_chat.py`, `conversation.py`, `modeling_intern_vit.py`, `modeling_internvl_chat.py`) are compared with the table of the files that were reviewed (no network,
  process or file-system calls) and printed at load; a different file stops the run.
- InternVL's own preprocessing: dynamic 448 px tiles, at most `VLM_MAX_TILES` (default 4) + a thumbnail, 256 image tokens per tile (a typical 640x480 COCO image is ONE tile = 256 tokens); fp16. Same protocol as the
  other VLMs (shared Vietnamese letter prompt in the model's chat template, one forward pass, letter-token logits, 2 option orders for choice, per-type temperature on calib, zero-shot + head-only 3 seeds on the full train).
- Needs `timm` (the InternViT code imports it): `run_vintern.sh` installs it if it is missing; setup.sh now installs it.
