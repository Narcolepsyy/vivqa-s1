"""Share of <unk> question / candidate tokens per split (words are built from train+calib only) + vocab sizes.
    python scripts/viso_vocab_stats.py --out output/viso/vocab_stats.json"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import data_utils.vocabs.s1_vocab  # noqa: E402,F401
from builders.vocab_builder import build_vocab  # noqa: E402
from data_utils.utils import preprocess_sentence  # noqa: E402
from train_viso import load_config  # noqa: E402
from utils.viso_s1 import read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default=os.path.join(REPO, "configs", "viso_vivqav2.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config_file, [])
    vocab = build_vocab(cfg.DATASET.VOCAB)
    res = {"vocab_size": len(vocab), "flat_answers": vocab.total_answers, "max_question_length": vocab.max_question_length}
    for split in ("train", "calib", "dev", "test"):
        qn = qu = cn = cu = 0
        for it in read_jsonl(os.path.join(cfg.DATASET.VOCAB.S1_DIR, f"{split}.jsonl")):
            q = it["questions"]["q"]
            toks = preprocess_sentence(q["instructions"], vocab.tokenizer)
            qn += len(toks); qu += sum(t not in vocab.stoi for t in toks)
            if q["type"] == "choice":
                for c in q["criteria"]:
                    ct = preprocess_sentence(c, vocab.tokenizer)[:6]
                    cn += len(ct); cu += sum(t not in vocab.stoi for t in ct)
        res[split] = {"question_tokens": qn, "question_unk_share": round(qu / max(qn, 1), 5),
                      "candidate_tokens": cn, "candidate_unk_share": round(cu / max(cn, 1), 5)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
