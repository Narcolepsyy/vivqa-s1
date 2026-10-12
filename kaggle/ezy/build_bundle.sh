#!/usr/bin/env bash
# Build the standalone folder the user copies to the rented machine: ezy/{setup.sh,run.sh,common.sh,extract_coco.py,pull_models.py,vlm_kernel.py,s1_vlm_eval.py,s1_vlm_lora.py,coco_images.py,README_EZY.md},
# ezy_bundle.tar.gz and ezy_paste.txt (a base64 block the user can paste into a terminal). Output: ~/data/ezy_bundle (or $1).
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HOME/data/ezy_bundle}"
mkdir -p "$OUT"
rm -rf "${OUT:?}/ezy" "$OUT/ezy_bundle.tar.gz" "$OUT/ezy_paste.txt"
mkdir -p "$OUT/ezy"
for f in setup.sh run.sh run_viso_ft.sh run_vintern.sh common.sh apply_update.sh extract_coco.py pull_models.py ezy_status.py overlay_files.txt README_EZY.md; do cp "$SRC/$f" "$OUT/ezy/"; done
cp "$SRC/../vlm_kernel.py" "$OUT/ezy/"
while IFS= read -r rel; do                                                   # the overlay tree: every file listed in overlay_files.txt, with its path under ViVQAv2-main
  [ -n "$rel" ] || continue
  mkdir -p "$OUT/ezy/overlay/$(dirname "$rel")"; cp "$SRC/../../$rel" "$OUT/ezy/overlay/$rel"
done < "$SRC/overlay_files.txt"
chmod +x "$OUT"/ezy/*.sh
( cd "$OUT" && tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 -czf ezy_bundle.tar.gz ezy )
SHA="$(sha256sum "$OUT/ezy_bundle.tar.gz" | awk '{print $1}')"
{
  echo "# Paste ALL of this into the container terminal (it creates ./ezy and verifies the checksum)."
  echo 'cd /workspace 2>/dev/null || cd ~'
  echo "base64 -d > ezy_bundle.tar.gz <<'EZY_EOF'"
  base64 -w 76 "$OUT/ezy_bundle.tar.gz"
  echo "EZY_EOF"
  echo "echo \"$SHA  ezy_bundle.tar.gz\" | sha256sum -c - && tar xzf ezy_bundle.tar.gz && cd ezy && ls"
} > "$OUT/ezy_paste.txt"
echo "bundle: $OUT ($(du -h "$OUT/ezy_bundle.tar.gz" | cut -f1) tar.gz, $(du -h "$OUT/ezy_paste.txt" | cut -f1) paste block), sha256 $SHA"
