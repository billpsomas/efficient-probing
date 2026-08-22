"""Evaluate trained probes against the ReImageNet re-annotations.

ReImageNet (Volkov, Kisel, Mishkina, Janouskova, Matas -- "Doomed to Re-Annotate,
Forever: The ImageNet Story", arXiv:2608.13783, CC BY 4.0) re-annotates the
ImageNet-1k VALIDATION set: multilabel corrections, an empty label set where no
ImageNet class is present, and pairs of classes declared equivalent. Training is
untouched, so every trained k-NN/LP/EP probe can be re-scored without re-training.

Two subcommands, so the GPU pass and the scoring stay independent:

  predict   run one trained head over the val set once and dump per-image
            predictions. Takes main_linprobe's own backbone flags (--model,
            --openclip, --finetune, --cls_features, ...) plus --head_ckpt, so the
            command for a row is its README command with the training flags
            dropped. Reuses build_backbone / build_probe_head / evaluate(), which
            forward --cls_features correctly (unlike main_linprobe --eval).

  score     torch-free. Match a predictions file against reannotation.jsonl and
            report, over the val images:
              original    top-1 against the original labels (the anchor: must
                          reproduce the head checkpoint's recorded accuracy)
              reimagenet  prediction counted correct if it is in the image's
                          reannotated label set, after mapping both sides through
                          the equivalent-class pairs; images with an empty label
                          set are excluded and counted separately

Predictions are stored as JSON {file_path: [pred, original_target]}, keyed the way
ReImageNet keys images ("n01440764/ILSVRC2012_val_00000293.JPEG"), so the two files
join on that string and nothing depends on ordering.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_predict(argv):
    import torch
    import main_linprobe as M
    from backbones import build_backbone
    from probe_heads import build_probe_head
    from engine_finetune import evaluate

    ap = argparse.ArgumentParser("predict", parents=[M.get_args_parser()])
    ap.add_argument("--head_ckpt", required=True,
                    help="a run's checkpoint-best.pth / checkpoint-.pth, or an "
                         "exported ep_head.pth from tools/export_ep_heads.py")
    ap.add_argument("--pred_out", required=True)
    args = ap.parse_args(argv)
    args.distributed = False
    device = torch.device(args.device)

    _, transform_val = M.build_transforms(args)
    _, dataset_val = M.build_datasets(args, transform_val, transform_val)
    loader = torch.utils.data.DataLoader(
        dataset_val, sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False)

    model = build_backbone(args, device)
    M.load_finetune_checkpoint(model, args)
    build_probe_head(model, args)

    ck = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict") or ck.get("model") or ck   # exported / raw / bare
    missing, unexpected = model.head.load_state_dict(sd, strict=True), None
    print("head loaded strict from %s (epoch %s, recorded acc %s)"
          % (args.head_ckpt, ck.get("epoch", ck.get("meta", {}).get("head_epoch", "?")),
             (ck.get("test_stats") or {}).get("test_acc1",
              ck.get("meta", {}).get("val_acc1_at_head_epoch", "?"))))
    model.to(device).eval()

    stats = evaluate(loader, model, device, return_targets_and_preds=True,
                     cls_features=args.cls_features, return_block=args.return_block)
    preds, targets = stats["preds"].tolist(), stats["targets"].tolist()
    assert len(preds) == len(dataset_val.samples)
    out = {}
    for (path, tgt), p, t in zip(dataset_val.samples, preds, targets):
        assert tgt == t, "SequentialSampler order broken -- refusing to mislabel"
        key = "/".join(path.replace("\\", "/").split("/")[-2:])   # synset/filename
        out[key] = [int(p), int(t)]
    with open(args.pred_out, "w") as fh:
        json.dump(out, fh)
    acc = 100.0 * sum(p == t for p, t in out.values()) / len(out)
    print("wrote %s: %d images, original top-1 %.2f  <- anchor, compare to the "
          "head checkpoint's recorded accuracy above" % (args.pred_out, len(out), acc))


def cmd_score(argv):
    ap = argparse.ArgumentParser("score")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--annotations", required=True,
                    help="directory holding reannotation.jsonl and class_update_config.json")
    args = ap.parse_args(argv)

    preds = json.load(open(args.preds))

    # equivalent classes score as one: map every class to its group representative
    canon = {}
    cfg_path = os.path.join(args.annotations, "class_update_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        pairs = cfg.get("equivalent_classes", cfg if isinstance(cfg, list) else [])
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                rep = min(int(x) for x in pair)
                for x in pair:
                    canon[int(x)] = rep
    c = lambda k: canon.get(k, k)

    n = orig_ok = re_ok = 0
    empty = missing = multilabel = 0
    for line in open(os.path.join(args.annotations, "reannotation.jsonl")):
        a = json.loads(line)
        rec = preds.get(a["file_path"])
        if rec is None:
            missing += 1
            continue
        p, t = rec
        labels = [int(x) for x in a["reannotated_labels"]]
        if not labels:
            empty += 1              # no ImageNet class present: excluded, counted
            continue
        n += 1
        if len(set(c(x) for x in labels)) > 1:
            multilabel += 1
        orig_ok += (c(p) == c(t))
        re_ok += (c(p) in set(c(x) for x in labels))

    print("images matched: %d scored + %d excluded (empty label set) + %d not in preds"
          % (n, empty, missing))
    print("multi-label among scored: %d (%.1f%%)" % (multilabel, 100.0 * multilabel / n))
    print("top-1 vs ORIGINAL labels (scored subset): %.2f" % (100.0 * orig_ok / n))
    print("top-1 vs REIMAGENET labels:               %.2f" % (100.0 * re_ok / n))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("predict", "score"):
        sys.exit("usage: eval_reimagenet.py {predict|score} ...")
    (cmd_predict if sys.argv[1] == "predict" else cmd_score)(sys.argv[2:])
