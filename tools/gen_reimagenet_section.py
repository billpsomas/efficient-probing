"""Build the README's ReImageNet Benchmark section from the scored predictions.

A cell is shown only when it cannot mislead: either the evaluated head is the
run's PEAK (so the RIN number pairs with the same head as the leaderboard's IN
number), or it is a final-epoch head within 0.25 of the peak (the late-peaking
group, where final and best are the same head for any practical purpose). Cells
whose head is further from the peak render as pending until that model is
recaptured -- showing them would compare a weaker head's RIN against a stronger
head's IN, which is exactly the artifact this rule exists to avoid.

Inputs (maintainer-local, not in the repo):
    ~/ep_release_notes/reimagenet_best_scores.txt      peak-head scores
    ~/ep_release_notes/reimagenet_lastepoch_table.txt  final-head scores
    <release>/manifest.json                            per-EP-head drift
Replaces the block between REIMAGENET:START/END markers in README.md.
"""
import csv, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.expanduser("~/ep_release_notes/reimagenet_best_scores.txt")
LAST = os.path.expanduser("~/ep_release_notes/reimagenet_lastepoch_table.txt")
MAN = "/scratch/project_465003083/psomasva/ep_heads_release/manifest.json"
START, END = "<!-- REIMAGENET:START -->", "<!-- REIMAGENET:END -->"
DRIFT_OK = 0.25


def log_drift(path):
    best = 0.0; last = None
    for l in open(path, errors="ignore"):
        m = re.match(r"^\d+, [\d.]+, [\d.]+, [\d.]+, ([\d.]+), ", l)
        if m:
            last = float(m.group(1)); best = max(best, last)
    return best - last if last is not None else 99.0


def main():
    rows = list(csv.DictReader(open(os.path.join(REPO, "results.csv"))))
    slug = lambda r: os.path.basename(os.path.dirname(r["ep_log"]))
    peak = {}
    for l in open(SC):
        m = re.match(r"(\S+) (LP|EP) top-1 vs REIMAGENET[^:]*:\s+([\d.]+)", l)
        if m: peak[(m.group(1), m.group(2))] = float(m.group(3))
    final = {}
    for l in open(LAST):
        p = l.strip().split(",")
        if len(p) == 5 and p[1] in ("LP", "EP"): final[(p[0], p[1])] = float(p[3])
    ep_drift = {}
    for h in json.load(open(MAN))["heads"]:
        ep_drift["%s %s" % (h["method"], h["arch"])] = \
            h["table_best_acc1"] - h["val_acc1_at_head_epoch"]

    table = []
    pend = 0
    for r in rows:
        nm = "%s %s" % (r["method"], r["arch"]); s = slug(r)
        cell = {}
        for arm, col in (("LP", "lp"), ("EP", "ep")):
            if (s, arm) in peak:
                cell[arm] = "%.2f" % peak[(s, arm)]
            elif (nm, arm) in final:
                d = ep_drift.get(nm, 99.0) if arm == "EP" else log_drift(r["lp_log"])
                if d <= DRIFT_OK:
                    cell[arm] = "%.2f" % final[(nm, arm)]
                else:
                    cell[arm] = "*pending*"; pend += 1
            else:
                cell[arm] = "*pending*"; pend += 1
        if "*pending*" in (cell["LP"], cell["EP"]):
            continue          # a row appears only once BOTH its heads qualify
        table.append((nm, float(r["lp"]), cell["LP"], float(r["ep"]), cell["EP"]))
    table.sort(key=lambda x: (-(float(x[4]) if x[4][0].isdigit() else -1), -x[3]))

    lines = [START,
        "<details>",
        '<summary><b>ReImageNet Benchmark</b> &mdash; the same probes under corrected labels</summary>', "",
        "[ReImageNet](https://huggingface.co/datasets/vrg-prague/ReImageNet) "
        "(Volkov, Kisel, Mishkina, Janou&scaron;kov&aacute;, Matas &mdash; "
        '*"Doomed to Re-Annotate, Forever: The ImageNet Story"*, '
        "[arXiv:2608.13783](https://arxiv.org/abs/2608.13783), CC BY 4.0) re-annotates the "
        "ImageNet-1k **validation set**: roughly 12% of the original labels are wrong, a third "
        "of images contain more than one labelable object, ~3% contain no ImageNet class at "
        "all, and 14 near-duplicate class pairs (laptop/notebook, sunglasses/sunglass, ...) "
        "are declared equivalent. Training is untouched, so every probe below is the **same "
        "trained head** as in the leaderboard, re-scored: a prediction counts if it names any "
        "class present in the image (after mapping equivalents), the 1,516 no-class images are "
        "excluded, leaving 48,484. Rerun with `tools/eval_reimagenet.py`; the annotations are "
        "gated on HuggingFace and not redistributed here.", "",
        "| model | LP (IN) | LP (RIN) | EP (IN) | EP (RIN) |",
        "|---|---:|---:|---:|---:|"]
    for nm, li, lr, ei, er in table:
        lines.append("| %s | %.2f | %s | %.2f | %s |" % (nm, li, lr, ei, er))
    lines += ["",
        "IN = the leaderboard's best-epoch top-1 on the full 50k val; RIN = ReImageNet "
        "multilabel top-1 on the 48,484-image subset, evaluated with the released head. A RIN "
        "row is shown only when both heads are the run's peak or within 0.25 of it, so the two "
        "columns always describe comparable heads; the remaining rows appear as their heads are "
        "recaptured. Two "
        "patterns worth reading off: the RIN&minus;IN gap **shrinks as encoders get stronger** "
        "(weak MIM probes gain 3+, frontier models 1.1&ndash;1.5 &mdash; strong probes track "
        "ImageNet's label errors more faithfully), and the top-5 ordering is the same under "
        "both label sets.", "",
        "</details>", END]

    p = os.path.join(REPO, "README.md"); t = open(p).read()
    block = "\n".join(lines)
    if START in t:
        t = t[:t.index(START)] + block + t[t.index(END) + len(END):]
    else:
        # first insertion: directly after the pooler section's closing details tag
        # anchor on the section's <summary>, not the first textual mention -- a
        # cross-reference link to it appears earlier and once misplaced the block
        anchor = t.index("</details>", t.index("<summary><b>What the encoder's own pooler scores")) + len("</details>")
        t = t[:anchor] + "\n\n" + block + t[anchor:]
    open(p, "w").write(t)
    print("ReImageNet section: %d rows, %d pending cells" % (len(table), pend))


if __name__ == "__main__":
    main()
