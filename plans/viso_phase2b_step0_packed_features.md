# Phase 2b, step 0: packed region features (planner: ie307-final-ec; implementor: ie307-final-7f)

Goal: replace 27 GB of per-image `.npy` dicts by ONE half-precision region-feature array (~5.2 GB) so training is not disk bound
and the data can be moved to Kaggle / a rented GPU. Phase 2a stays accepted; this step must not change its default behaviour.

## Measured on 400 random files (planner, for reference; recompute for all images)
- Images needed by S1 (train+calib+dev+test): 34,893. Regions per image: min 10, mean 36.5, max 83. Max |value| about 121.
- fp16 round-trip max relative error about 4.6e-4 (no overflow). Exact-length storage ~5.2 GB; padding to 83 would be 11.9 GB: do NOT pad.

## Hard rules
- READ-ONLY: `../vivqav2_dataset` (incl. vinvl_vinvl), `vivqa_s1/`, all pre-existing repo files. NEW files only, except that you may edit your OWN Phase 2a files
  (`data_utils/datasets/s1_dataset.py`, `configs/viso*.yaml`, `docs/viso_README.md`, tests) to add the option below. Keep defaults unchanged
  (`FEATURE_FORMAT: npy`) so the accepted 46 tests and tiny runs still behave identically.
- Output goes OUTSIDE the project folder (the project is zipped for submission): default `~/data/vivqa_packed/`. Never inside the repo, vivqa_s1 or vivqav2_dataset.
- No network. Be gentle with the machine: run the pack under `nice -n 10 ionice -c3`, stream image by image, keep RAM use under ~1 GB.
- Deterministic: images sorted by coco_id; rerunning gives byte-identical files (sha256 equal).

## Files to create
```
scripts/viso_pack_features.py      pack + --verify
utils/viso_packed_features.py      PackedFeatures reader (lazy memmap, fork/worker safe)
tests/viso/test_packed_features.py
```
Edit (own files only): `data_utils/datasets/s1_dataset.py` (+ config key `DATASET.FEATURE_DATASET.FEATURE_FORMAT: npy|packed`, `PACKED_PATH`), `docs/viso_README.md`.

## Pack format (directory `PACKED_PATH`)
- `regions_fp16.bin`: raw little-endian float16, shape [total_regions, 2048], images concatenated in ascending coco_id order, exact lengths.
- `index.npz`: `coco_id` (int64, sorted), `offset` (int64, first row), `n_regions` (int32).
- `tags.jsonl`: one line per coco_id (same order): {"coco_id", "objects": [...], "attributes": [[...],...]} copied from the npy dict (small; reserved for a later state_text ablation).
- `meta.json`: dtype, dim=2048, n_images, total_regions, source dir, list of split files used, sha256 of regions_fp16.bin / index.npz / tags.jsonl, creation date, script git-less version string.
- The image set = union of `coco_id` over vivqa_s1/data/{train,calib,dev,test}.jsonl. Fail loudly (list ids) if a source `.npy` is missing or has region_features of a different dim/dtype.

## Reader and dataset integration
- `PackedFeatures(path)`: opens `np.memmap` lazily on first access (so DataLoader workers each open their own), `get(coco_id)` returns `region_features` as a float32 torch tensor of shape (n,2048) (cast from fp16), plus `n_regions`. Unknown id -> KeyError with the id.
- `S1FeatureDataset` with `FEATURE_FORMAT: packed` uses it instead of `np.load`; the collate/mask logic is unchanged (explicit `region_mask`).
- The `npy` path stays the default and untouched.

## Verification (script `--verify`, writes `verify.json` next to the pack; quote it in your report)
1. Counts: n_images == |union of S1 coco_ids|; sum(n_regions) == total_regions == file size / (2*2048).
2. Spot check: 1,000 random ids (seed 0) packed vs original npy: identical n_regions, max abs diff <= 1e-3 * max|original| per image; report the worst case.
3. Sample of 100 ids: also check the objects/attributes in tags.jsonl equal the npy ones.
4. Equivalence: ViSO and FlatS1 forward on the same 8 real items, npy vs packed: max abs diff of the fused vector h <= 5e-3 (document the value).
5. Timing (report, cold cache not required): load 2,000 random items packed vs npy, items/s each.
6. Determinism: pack twice into two temp dirs; the three sha256 are equal (you may restrict the second run to a 500-image subset via `--limit 500` and compare that).

## Tests (pytest, CPU, fast)
Build a pack from 20 real ids in a tmp dir; reader returns the original arrays within fp16 tolerance; offsets/n_regions consistent; missing id raises KeyError; `S1FeatureDataset` with FEATURE_FORMAT=packed yields the same shapes/masks as npy for those ids; the default config still uses npy; worker-safe (num_workers=2 returns data). The old 46 tests must still pass.

## Acceptance
1. All 34,893 images packed; verify.json produced and passing all checks above.
2. `pytest tests/viso -p no:cacheprovider` (old + new) passes.
3. No pre-existing file modified (sha256 manifest like before; only your own Phase 2a files and the new ones differ); the pack lives outside the project folder.
4. Report: sizes (quote meta.json and verify.json), time to pack, the timing comparison, deviations. Then stop; do not start training.
