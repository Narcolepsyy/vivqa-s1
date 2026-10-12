"""Bridge to the frozen ViVQA-S1 benchmark (../vivqa_s1_v2, read-only): makes `s1bench` importable (appended to
sys.path so it can never shadow this repo's own packages) and exposes the S1 constants used by ViSO."""
import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1_ROOT = os.environ.get("VIVQA_S1_ROOT") or os.path.abspath(os.path.join(REPO_ROOT, "..", "vivqa_s1_v2"))
if S1_ROOT not in sys.path:
    sys.path.append(S1_ROOT)

from s1bench.schema import label_space  # noqa: E402  (after the sys.path tweak)

TYPE_IDS = {"noul": 0, "score": 1, "choice": 2}
TYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}
MAX_SPACE = 16                     # widest label space (choice items have exactly 16 candidates)
SPACE_SIZE = {"noul": 2, "score": 11}
NOUL_KEYS = ["false", "true"]
SCORE_KEYS = [str(i) for i in range(11)]
NEG_INF = -1e9                     # masked logits (fp32)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def yesno_form(it):
    """question form of a yes/no item: existence ('co ... nao', 'is there / are there ...'), attribute ('mau' = colour, 'dang' = ongoing action) or other (used by the learning-curve subsets and the
    yes/no-by-form reports)"""
    vi = it["questions"]["q"]["instructions"].lower()
    en = (it.get("meta", {}).get("en_q") or "").lower()
    if re.search(r"\bcó\b.*\bnào\b", vi) or en.startswith(("is there", "are there")):
        return "existence"
    if "màu" in vi or re.search(r"\bđang\b", vi):
        return "attribute"
    return "other"
