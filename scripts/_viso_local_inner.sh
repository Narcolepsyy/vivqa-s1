#!/usr/bin/env bash
cd "/home/khaitran/project/IE307-Final/ViVQAv2-main"
exec /home/khaitran/.venvs/siglip2-gpu/bin/python -B "/home/khaitran/project/IE307-Final/ViVQAv2-main/scripts/viso_kaggle_run.py" --experiments "/home/khaitran/project/IE307-Final/ViVQAv2-main/kaggle/experiments_siglip2.json" --out "/home/khaitran/data/viso_runs_v2" --code-root "/home/khaitran/project/IE307-Final/ViVQAv2-main" \
  --s1-dir "/home/khaitran/project/IE307-Final/ViVQAv2-main/../vivqa_s1_v2" --packed "/home/khaitran/data/siglip2_packed" --gpu 0 --deadline-hours 100000
