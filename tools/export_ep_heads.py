"""Repackage the trained EP heads for public release.

The raw training checkpoints cannot be published as they are, for two reasons this
script exists to fix:

  * they embed the full argparse Namespace, whose data_path / output_dir /
    finetune values are absolute cluster paths -- the same leak already scrubbed
    from the published logs;
  * they carry the LARS optimizer state and grad-scaler state, which roughly
    doubles the file size and serves no one downstream.

Each exported file keeps only the head's state_dict plus honest metadata. The
metadata states plainly that the head is the FINAL epoch of its run, not the best:
save_model overwrites one file per epoch, so for encoders that peak early (every
VLM here) the saved head scores below the table's best-epoch number. Both figures
are recorded per head, so nothing has to be taken on trust.

Selection mirrors the leaderboard: for every row of results.csv, the run whose
variant, best accuracy AND head size all match the published EP log. Two rows need
explicit pinning (EVA02-CLIP E-14, whose local log a pre-fix resume destroyed, and
SigLIP2 SO400M, where two runs share an identical peak); DiT-XL/2 has no surviving
checkpoint and is recorded as missing rather than silently dropped.

Usage:  python tools/export_ep_heads.py --out DIR
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import torch

PIN = {
    ("EVA02-CLIP", "E-14"): "outputs/imagenet/vit_e/linprobe_eva02_e14_ep_q32_d1_imagenet1k",
    ("SigLIP2", "SO400M/14"): "outputs/imagenet/so400m/linprobe_siglip2_so400m_ep_q32_d1_imagenet1k",
}
MISSING = {
    ("DiT", "DiT-XL/2"): "no surviving checkpoint: the run's output directory was deleted "
                         "after its logs were published; re-run required to export a head",
}


def loginfo(path):
    rep = None; best = 0.0; n = 0; params = None
    for l in open(path, errors="ignore"):
        if l.startswith("Representation:"):
            rep = l.split(":", 1)[1].strip()
        if l.startswith("Trainable Parameters"):
            params = l.split(":")[1].strip()
        m = re.match(r"^\d+, [\d.]+, [\d.]+, [\d.]+, ([\d.]+), ", l)
        if m:
            n += 1; best = max(best, float(m.group(1)))
    return rep, best, n, params


def find_run(r, dirs):
    key = (r["method"], r["arch"])
    if key in PIN:
        return PIN[key]
    var = r["ep_variant"].strip() or "ep"
    want = float(r["ep"])
    pub = loginfo(r["ep_log"])
    cand = [(d, i) for d, i in dirs.items()
            if i[0] == var and abs(i[1] - want) < 0.005 and i[3] == pub[3]]
    if len(cand) > 1:
        cand.sort(key=lambda x: -x[1][2])
        cand = [c for c in cand if c[1][2] == cand[0][1][2]]
    if len(cand) != 1:
        raise SystemExit("cannot pin %s %s: %d candidates" % (r["method"], r["arch"], len(cand)))
    return cand[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dirs = {}
    for d in glob.glob("outputs/imagenet/*/linprobe_*"):
        p = os.path.join(d, "training_log.txt")
        if os.path.isdir(d) and os.path.exists(p):
            dirs[d] = loginfo(p)

    manifest = {"heads": [], "missing": []}
    for r in csv.DictReader(open("results.csv")):
        key = (r["method"], r["arch"])
        slug = os.path.basename(os.path.dirname(r["ep_log"]))
        if key in MISSING:
            manifest["missing"].append({"method": r["method"], "arch": r["arch"],
                                        "reason": MISSING[key]})
            print("SKIP  %-30s %s" % (" ".join(key), MISSING[key][:60]))
            continue
        run = find_run(r, dirs)
        # runs trained after the save-best change carry the peak separately; use it
        cks = glob.glob(run + "/checkpoint-best.pth") or \
              glob.glob(run + "/checkpoint*.pth")
        assert len(cks) == 1, (key, cks)
        is_peak = cks[0].endswith("checkpoint-best.pth")
        ck = torch.load(cks[0], map_location="cpu", weights_only=False)
        sd = ck["model"]
        meta = {
            "method": r["method"], "arch": r["arch"], "pretrain": r["pretrain"],
            "image_size": int(r["image_size"]),
            "model": r["model"], "loader": r["loader"],
            "pretrain_tag": r["pretrain_tag"], "finetune": r["finetune"],
            "extra_flags": r.get("extra_flags", ""),
            "cls_features": r["ep_variant"].strip() or "ep",
            "ep_queries": int(r["ep_queries"] or 32), "d_out": 1,
            "head_epoch": int(ck["epoch"]),
            "val_acc1_at_head_epoch": round(float(ck["test_stats"]["test_acc1"]), 2),
            "table_best_acc1": float(r["ep"]),
            "note": ("best-epoch head: saved at the run's peak, the number the table reports."
                     if is_peak else
                     "final-epoch head, not best-epoch: checkpoints overwrite per epoch. "
                     "table_best_acc1 is the peak of the same run; the difference is the "
                     "post-peak decline of a frozen-feature probe, largest on VLM encoders."),
            "training_log": r["ep_log"],
        }
        # weights under their own key -- meta also has a "model" (the --model string),
        # and flattening the two into one dict silently replaced the state_dict with it
        out = {"state_dict": sd, "meta": meta}
        # the release must carry no trace of the training cluster
        assert "/scratch/" not in repr(meta) and "/projappl/" not in repr(meta), meta
        dst = os.path.join(a.out, slug)
        os.makedirs(dst, exist_ok=True)
        torch.save(out, os.path.join(dst, "ep_head.pth"))
        json.dump(meta, open(os.path.join(dst, "config.json"), "w"), indent=1)
        n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        manifest["heads"].append({**meta, "params_incl_bn_stats": n,
                                  "file": slug + "/ep_head.pth",
                                  "size_mb": round(os.path.getsize(os.path.join(dst, "ep_head.pth")) / 1e6, 1)})
        print("OK    %-30s ep%-3d acc %.2f (table %.2f)  %s" %
              (" ".join(key), meta["head_epoch"], meta["val_acc1_at_head_epoch"],
               meta["table_best_acc1"], slug))
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    tot = sum(h["size_mb"] for h in manifest["heads"])
    print("\nexported %d heads (%.0f MB), %d missing -> %s"
          % (len(manifest["heads"]), tot, len(manifest["missing"]), a.out))


if __name__ == "__main__":
    main()
