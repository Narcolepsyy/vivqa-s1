#!/usr/bin/env bash
# Local ViSO matrix (15 jobs of kaggle/experiments_siglip2.json) on the laptop GPU: DETACHED (tmux), sleep-inhibited, sequential, resumable.
#   start / resume (same command):  bash scripts/launch_viso_local.sh
#   watch:   tmux attach -t viso        (detach with Ctrl-b d)      tail -f ~/data/viso_runs_v2/train.log      tail -f ~/data/viso_runs_v2/gpu.log
#   stop:    tmux kill-session -t viso  (a running job resumes from its last checkpoint on the next start)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HOME/data/viso_runs_v2"; BACKUP="$HOME/data/viso_runs_v2_backup"
PY="$HOME/.venvs/siglip2-gpu/bin/python"
if tmux has-session -t viso 2>/dev/null; then echo "tmux session 'viso' already running: tmux attach -t viso"; exit 0; fi
mkdir -p "$OUT" "$BACKUP"
# the runner itself (own script: systemd-inhibit needs a plain executable, and a relative path would depend on tmux's working dir)
INNER="$REPO/scripts/_viso_local_inner.sh"
cat > "$INNER" <<EOT
#!/usr/bin/env bash
cd "$REPO"
exec $PY -B "$REPO/scripts/viso_kaggle_run.py" --experiments "$REPO/kaggle/experiments_siglip2.json" --out "$OUT" --code-root "$REPO" \\
  --s1-dir "$REPO/../vivqa_s1_v2" --packed "$HOME/data/siglip2_packed" --gpu 0 --deadline-hours 100000
EOT
chmod +x "$INNER"
# window 0: the runner (never changes configs; priority order = manifest order; no deadline; GPU 0)
tmux new-session -d -s viso -n train "systemd-inhibit --what=sleep:idle:handle-lid-switch --why='ViSO training' bash $INNER 2>&1 | tee -a $OUT/train.log"
# window 1: GPU log every 5 minutes
tmux new-window -t viso -n gpu "while true; do echo \"\$(date -Is),\$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader)\" >> $OUT/gpu.log; sleep 300; done"
# window 2: copy each finished run (DONE.json present) to the backup dir, every 10 minutes
tmux new-window -t viso -n backup "while true; do for d in $OUT/runs/*/; do j=\$(basename \$d); if [ -f \$d/DONE.json ] && [ ! -f $BACKUP/\$j/.complete ] && [ -f \$d/post_done.json ]; then mkdir -p $BACKUP/\$j && cp -r \$d/. $BACKUP/\$j/ && touch $BACKUP/\$j/.complete && echo \"\$(date -Is) backed up \$j\" >> $OUT/backup.log; fi; done; sleep 600; done"
echo "started. tmux attach -t viso | tail -f $OUT/train.log | tail -f $OUT/gpu.log"
