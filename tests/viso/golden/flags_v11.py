"""Sidecar evaluation flags (v1.1 release). Items, golds and candidates are never edited; test views are defined here.

Flags (per item, written to data/eval_flags.jsonl by scripts/09_eval_flags.py):
  text_answer_llm        all splits: an LLM (DeepSeek) judged the gold to be a Vietnamese translation of text that should stay
                         as written (informational; never excludes anything)
  audit_gold_conflict    test: label-audit kinds a_majority_disagrees / b_ngoai_for_off (rule-based, 10 VQA v2 answers)
  audit_orange_fruit     test: label-audit kind b_orange_fruit_for_color (reported only, stays in Core)
  image_seen             test: the item's image also appears in train, calib or dev
  text_answer_confirmed  test: a human marked the gold as a translation of text that should stay as written
  gold_wrong_human       test: a human marked the gold wrong (human_gold_correct == 0)
  human_unsure / human_multi_correct   reported only
Views of the test split:
  All           every item (comparable with the ViVQAv2 lineage)
  Core          All minus CORE_EXCLUDE   (headline)
  Image-unseen  Core minus image_seen    (diagnostic)
  S1-Human      the existing human_subset block of every view"""
import csv
import io
import json
import re
from collections import defaultdict

CORE_EXCLUDE = ("audit_gold_conflict", "text_answer_confirmed", "gold_wrong_human")
UNSEEN_EXTRA = ("image_seen",)
AUDIT_EXCLUDE_KINDS = ("a_majority_disagrees", "b_ngoai_for_off")
AUDIT_REPORT_KINDS = {"b_orange_fruit_for_color": "audit_orange_fruit"}


def parse_audit_md(text):
    """rows of data/label_audit_test.md -> [(id, kind, evidence)] (table rows: | id | kind | question | gold | answers | suggested |)"""
    out = []
    for line in text.splitlines():
        p = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(p) >= 6 and p[0].isdigit() and re.match(r"[ab]_", p[1]):
            out.append((int(p[0]), p[1], {"gold": p[3], "answers_10": p[4], "suggested": p[5]}))
    return out


def read_human_csv(text):
    """Tolerant reader for the human sheets (utf-8-sig, ',' or ';' separated)."""
    text = text.lstrip("﻿")
    head = text.split("\n", 1)[0]
    delim = ";" if head.count(";") > head.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    for r in rows:
        for k in list(r):
            r[k] = (r[k] or "").strip()
    return rows


def _norm_v(v):
    return {"1.0": "1", "0.0": "0"}.get(v, v)


def build_flags(items_by_split, text_items, audit_rows, human_rows):
    """Pure function. items_by_split: {split: [item]}; text_items: [{"id","split",...}]; audit_rows from parse_audit_md;
    human_rows: list of dict rows of the human sheets (H100 and ADJ). Returns [{"id","split","flags","evidence"}] sorted by id."""
    split_of = {it["id"]: sp for sp, its in items_by_split.items() for it in its}
    flags = defaultdict(lambda: ([], {}))

    def add(i, flag, ev=None):
        if i not in split_of:
            raise KeyError(f"flag {flag} for unknown id {i}")
        fl, evd = flags[i]
        if flag not in fl:
            fl.append(flag)
        if ev:
            evd.setdefault(flag, ev)

    for t in text_items:
        add(int(t["id"]), "text_answer_llm", {k: t.get(k, "") for k in ("visible_text", "status", "reason")})
    for i, kind, ev in audit_rows:
        if kind in AUDIT_EXCLUDE_KINDS:
            add(i, "audit_gold_conflict", {"kind": kind, **ev})
        elif kind in AUDIT_REPORT_KINDS:
            add(i, AUDIT_REPORT_KINDS[kind], {"kind": kind, **ev})
    seen = {sp: {it["image"] for it in its} for sp, its in items_by_split.items()}
    for it in items_by_split.get("test", []):
        where = [sp for sp in ("train", "calib", "dev") if it["image"] in seen.get(sp, ())]
        if where:
            add(it["id"], "image_seen", {"in": where})
    for r in human_rows:
        if not r.get("id", "").isdigit() or int(r["id"]) not in split_of or split_of[int(r["id"])] != "test":
            continue
        i, v = int(r["id"]), _norm_v(r.get("human_gold_correct", ""))
        ev = {"sheet": r.get("_sheet", ""), "notes": r.get("human_notes", ""), "human_correct_candidates": r.get("human_correct_candidates", "")}
        # union over the sheets: a '0' in EITHER sheet flags the item; '?' and blanks never exclude
        if v == "0":
            add(i, "gold_wrong_human", ev)
        elif v == "?":
            add(i, "human_unsure")
        # text confirmation: only ADJ rows of check_kind text (rows without a check_kind column are text rows by definition)
        if _norm_v(r.get("human_text_translated", "")) == "1" and r.get("check_kind", "text") == "text":
            add(i, "text_answer_confirmed", ev)
        try:
            if int(r.get("human_n_correct") or 0) >= 2:
                add(i, "human_multi_correct")
        except ValueError:
            pass
    return [{"id": i, "split": split_of[i], "flags": sorted(fl), "evidence": {k: ev[k] for k in sorted(ev)}}
            for i, (fl, ev) in sorted(flags.items())]


def load_flags(path):
    """-> {id: set(flags)}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = set(r["flags"])
    return out


def split_views(gold_items, flags):
    """-> {"all": [...], "core": [...], "image_unseen": [...]}, plus excluded counts by flag."""
    core = [it for it in gold_items if not (flags.get(it["id"], set()) & set(CORE_EXCLUDE))]
    unseen = [it for it in core if not (flags.get(it["id"], set()) & set(UNSEEN_EXTRA))]
    exc = {"core": {f: sum(f in flags.get(it["id"], ()) for it in gold_items) for f in CORE_EXCLUDE},
           "image_unseen": {f: sum(f in flags.get(it["id"], ()) for it in core) for f in UNSEEN_EXTRA}}
    return {"all": list(gold_items), "core": core, "image_unseen": unseen}, exc
