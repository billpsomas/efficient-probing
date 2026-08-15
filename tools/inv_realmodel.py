"""Fingerprint a REAL encoder + real checkpoint + real probe head, end to end.

tools/inv_heads.py proves the head registry is faithful, but only against a stub
encoder exposing `.head` and `.patch_embed.num_patches`. This closes that blind
spot: it builds an actual backbone, runs the actual --finetune checkpoint load, and
builds the actual probe head, then fingerprints the result.

It deliberately covers what training-based comparison cannot do cheaply:

  * the probe head's INITIALISED WEIGHTS after the real construction order, so a
    changed number of RNG draws anywhere upstream shows up as a different sha256
  * which parameters ended up trainable (the freeze step)
  * the encoder's own weights after the checkpoint load, so a silently-unloaded
    tensor shows up

The same file runs in both trees: in the pre-refactor tree it execs the relevant
source regions out of main_linprobe.py; in the refactored tree it calls the
extracted functions. Everything happens on CPU, so this needs no allocation.

Usage:  python tools/inv_realmodel.py --out FILE [--finetune CKPT]
"""
import argparse
import hashlib
import json
import os
import sys

import torch


def sha_params(named):
    h = hashlib.sha256()
    for n, p in sorted(named):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def load_source_region(src, start_marker, end_marker):
    text = open(src).read()
    a = text.index(start_marker)
    b = text.index(end_marker, a)
    body = text[a:b]
    return "".join((l[4:] if l.startswith("    ") else l) + "\n" for l in body.split("\n"))


def build_original(args, device):
    """Replay the pre-refactor if/elif chains out of main_linprobe.py source."""
    import main_linprobe as M

    ns = dict(M.__dict__)
    ns.update(dict(args=args, device=device, torch=torch, Path=__import__("pathlib").Path))

    chain = load_source_region("main_linprobe.py",
                               '    if args.model.startswith("capi"):',
                               "    # NOTE: --knn_eval used to be excluded here")
    exec(chain, ns)
    model = ns["model"]

    ckpt = load_source_region("main_linprobe.py",
                              "    # NOTE: --knn_eval used to be excluded here",
                              '    if args.cls_features == "abmilp"')
    ns["model"] = model
    exec(ckpt, ns)

    head = load_source_region("main_linprobe.py",
                              '    if args.cls_features == "abmilp"',
                              "    if args.finetuning:")
    exec(head, ns)

    freeze = load_source_region("main_linprobe.py",
                                "    if args.finetuning:\n        # unfreeze all",
                                "    model.to(device)")
    exec(freeze, ns)
    return ns["model"]


def build_refactored(args, device):
    from backbones import build_backbone
    from probe_heads import build_probe_head
    import main_linprobe as M

    model = build_backbone(args, device)
    M.load_finetune_checkpoint(model, args)
    build_probe_head(model, args)
    M.set_trainable(model, args)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--finetune", default="")
    ap.add_argument("--group", choices=["vit", "wrappers"], default="vit")
    a = ap.parse_args()

    sys.path.insert(0, ".")
    import main_linprobe as M

    refactored = os.path.exists("backbones.py")
    build = build_refactored if refactored else build_original
    print("tree: %s" % ("REFACTORED" if refactored else "ORIGINAL"))

    # Each case pins one construction path. The "vit" group runs the models_vit
    # fallback (needs the `ep` venv, timm 0.9.16); the "wrappers" group covers five
    # DIFFERENT builders in backbones.py -- two prefix-dispatched, three flag-
    # dispatched -- because that file's bodies were transcribed by hand and only its
    # dispatch ORDER is otherwise verified. All use already-cached weights.
    HUB = "/scratch/project_465003083/psomasva/cache/torch/hub/checkpoints"
    GROUPS = {
        "vit": [
            ("vit_base_patch16", "ep", {}),
            ("vit_base_patch16", "cls", {}),
            ("vit_base_patch16", "ep_all", {}),
            ("vit_base_patch16", "simpool", {}),
        ],
        "wrappers": [
            ("dinov2_vitb14", "cls", {}),
            ("capi_vitl14_in1k", "cls", {}),
            ("hiera_base_224.mae", "cls", {"timm": True}),
            ("ViT-L-16-SigLIP-256", "cls", {"openclip": True, "openclip_pretrain": "webli"}),
            ("dinov3_vitb16", "cls",
             {"dinov3_weights": HUB + "/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"}),
        ],
    }

    out = {}
    for model_name, feat, overrides in GROUPS[a.group]:
        key = "%s/%s" % (model_name, feat)
        args = M.get_args_parser().parse_args([])
        args.model = model_name
        args.cls_features = feat
        args.nb_classes = 1000
        args.finetune = a.finetune if not overrides else ""
        args.eval = False
        args.knn_eval = False
        for k, v in overrides.items():
            setattr(args, k, v)
        try:
            torch.manual_seed(0)
            model = build(args, torch.device("cpu"))
            trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
            out[key] = {
                "head_repr": " ".join(repr(model.head).split()),
                "head_sha": sha_params(model.head.named_parameters()),
                "n_trainable": sum(p.numel() for _, p in trainable),
                "n_frozen": sum(p.numel() for _, p in frozen),
                "trainable_names": sorted(n for n, _ in trainable),
                "backbone_sha": sha_params(frozen),
            }
            print("  %-28s head_sha=%s trainable=%d" % (key, out[key]["head_sha"][:16], out[key]["n_trainable"]))
        except Exception as e:
            out[key] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
            print("  %-28s ERROR %s" % (key, out[key]["error"][:110]))

    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
