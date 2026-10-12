"""Paired bootstrap between two runs from their per-item dev dumps (viso_vision_check.py --dump-peritem): the same dev items, per type and for the yes/no question forms:
mean (A - B) of accuracy and of NLL with a 95% CI over items (2000 resamples). A positive accuracy difference / negative NLL difference favours A.
    python scripts/paired_bootstrap.py --a peritem_KD5_all.npz --b peritem_TSIG_all.npz [--label-a KD5 --label-b TSIG]"""
import argparse

import numpy as np

TYPES = ("noul", "score", "choice")


def load(p):
    z = np.load(p, allow_pickle=True)
    return {int(i): (int(t), float(c), float(n), str(f)) for i, t, c, n, f in zip(z["ids"], z["type_id"], z["correct"], z["nll"], z["form"])}


def ci(d, rng, n_boot=2000):
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(a, b, seed=0):
    ids = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    rows = []
    groups = {"all": ids}
    for t, name in enumerate(TYPES):
        groups[name] = [i for i in ids if a[i][0] == t]
    for f in ("existence", "attribute", "other"):
        groups["yes/no: " + f] = [i for i in ids if a[i][0] == 0 and a[i][3] == f]
    for g, sel in groups.items():
        if not sel:
            continue
        dacc = np.array([a[i][1] - b[i][1] for i in sel])
        dnll = np.array([a[i][2] - b[i][2] for i in sel])
        rows.append((g, len(sel), ci(dacc, rng), ci(dnll, rng)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    x = ap.parse_args(argv)
    rows = compare(load(x.a), load(x.b))
    print(f"paired bootstrap over dev items, {x.label_a} - {x.label_b} (positive accuracy diff / negative NLL diff favours {x.label_a}); 95% CI")
    print("| group | n | acc diff [CI] | NLL diff [CI] |\n|---|---|---|---|")
    for g, n, (da, la, ha), (dn, ln, hn) in rows:
        sig_a = "*" if la > 0 or ha < 0 else ""
        sig_n = "*" if ln > 0 or hn < 0 else ""
        print(f"| {g} | {n} | {da:+.4f} [{la:+.4f}, {ha:+.4f}]{sig_a} | {dn:+.4f} [{ln:+.4f}, {hn:+.4f}]{sig_n} |")
    return rows


if __name__ == "__main__":
    main()
