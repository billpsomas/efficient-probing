"""Fingerprint the pooling-head construction for every --cls_features value.

The head chain is the riskiest part of the refactor: 14 poolings, each wrapping the
classifier in Sequential(pool, BatchNorm1d, head), built *after* the seed is set, so
the module order also fixes the RNG draw order. This records, per value:

  * the module structure (repr of the Sequential)
  * every parameter's name and shape
  * a sha256 over the initialised weights under a fixed seed

The third one is what catches a reordered constructor: same structure, same shapes,
different draws. Runs against a stub encoder, so it needs no GPU, no network and no
data -- which is the point, since it has to be runnable while the cluster is busy.

Usage:  python tools/inv_heads.py --mode {baseline,check} --out FILE [--src FILE]
"""
import argparse
import hashlib
import json
import re
import sys

import torch
import torch.nn as nn


def install_cuda_shim():
    """Let CUDA-only poolings build on CPU, so they can actually be fingerprinted.

    SimPool and SimPool_nolinears allocate torch.tensor(..., device='cuda') in
    __init__ (poolings/simpool.py:21,105), so on a login node they raise before any
    weight exists -- which silently reduced 24 of this file's configs to comparing an
    error string instead of comparing weights. Redirecting those allocations to CPU
    turns them back into real comparisons. The redirected tensors are constants
    (eps, gamma), not parameters, so they do not enter the fingerprint anyway.
    """
    real_tensor = torch.tensor

    def tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    torch.tensor = tensor


class StubPatchEmbed(nn.Module):
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches = num_patches


class StubEncoder(nn.Module):
    """Only the attributes the head chain actually reads."""

    def __init__(self, dim, nb_classes, num_patches):
        super().__init__()
        self.head = nn.Linear(dim, nb_classes)
        self.patch_embed = StubPatchEmbed(num_patches)
        self.embed_dim = dim


class StubArgs:
    pass


_DEFAULTS = None


def make_args(cls_features, dim, nb_classes, model_name):
    """A namespace carrying the REAL parser defaults, not hand-copied ones.

    Taking these from get_args_parser() rather than transcribing them is what keeps
    the fingerprints representative: the abmilp flags in particular default to
    sa="both", depth=2, content="all", which is not what you would guess.
    """
    global _DEFAULTS
    if _DEFAULTS is None:
        import main_linprobe
        _DEFAULTS = vars(main_linprobe.get_args_parser().parse_args([]))
    a = StubArgs()
    for k, v in _DEFAULTS.items():
        setattr(a, k, v)
    a.cls_features = cls_features
    a.nb_classes = nb_classes
    a.model = model_name
    return a


def extract_head_chain(src_path):
    """Pull the head-construction region out of the original main_linprobe.py."""
    src = open(src_path).read()
    start = src.index('    if args.cls_features == "abmilp"')
    end = src.index("    if args.finetuning:", start)
    body = src[start:end]
    # dedent one level so it can exec at module scope
    return "".join(
        (line[4:] if line.startswith("    ") else line) + "\n" for line in body.split("\n")
    )


def fingerprint(build, cls_features, dim, nb_classes, model_name, num_patches):
    torch.manual_seed(0)
    model = StubEncoder(dim, nb_classes, num_patches)
    try:
        build(model, make_args(cls_features, dim, nb_classes, model_name))
    except Exception as e:  # a branch that raises today must raise after, too
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    head = model.head
    params = [(n, tuple(p.shape)) for n, p in sorted(head.named_parameters())]
    h = hashlib.sha256()
    for n, p in sorted(head.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return {
        "repr": re.sub(r"\s+", " ", repr(head)),
        "params": params,
        "n_params": sum(p.numel() for p in head.parameters()),
        "sha256": h.hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "check"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", help="main_linprobe.py to exec the chain from (baseline mode)")
    ap.add_argument("--cuda_shim", action="store_true",
                    help="Redirect device='cuda' allocations to CPU so SimPool/eSimPool can build")
    args = ap.parse_args()
    if args.cuda_shim:
        install_cuda_shim()

    sys.path.insert(0, ".")
    from models_vit import CLS_FT_CHOICES

    if args.mode == "baseline":
        chain = extract_head_chain(args.src)
        ns = {}
        exec(
            "import torch\n"
            "from poolings.abmilp import ABMILPHead\n"
            "from poolings.simpool import SimPool, SimPool_nolinears\n"
            "from poolings.clip.attention_pool import AttentionPoolLatent\n"
            "from poolings.clip.attention_pool2d import AttentionPool2d\n"
            "from poolings.jepa.attentive_pooler import AttentivePooler\n"
            "from poolings.aim import AttentionPoolingClassifier\n"
            "from poolings.cbam import CbamPooling\n"
            "from poolings.coca_pytorch import CrossAttention as CocaPooling\n"
            "from poolings.other_pool import CAPooling, DinoViTBlockPooling\n"
            "from poolings.dolg.dolg import SpatialAttention2d\n"
            "from poolings.cae_att import CAEAttentiveBlock\n"
            "from poolings.ep import EfficientProbing\n"
            "def build(model, args):\n" + "".join("    " + l + "\n" for l in chain.split("\n")),
            ns,
        )
        build = ns["build"]
    else:
        from probe_heads import build_probe_head as build

    # widths spanning the published rows: ViT-S/B/L/H/g plus SigLIP's 1152
    results = {}
    for cls_features in sorted(set(CLS_FT_CHOICES) | {"cls", "gap", "pos", "raw"}):
        for dim in (384, 768, 1024, 1152, 1280, 1536):
            key = "%s@%d" % (cls_features, dim)
            results[key] = fingerprint(build, cls_features, dim, 1000, "vit_base_patch16", 196)
    # the two model-name- and d_out-sensitive branches
    for model_name in ("capi_vitl14_in1k", "vit_base_patch16"):
        results["clip@1024/%s" % model_name] = fingerprint(
            build, "clip", 1024, 1000, model_name, 196
        )
    for d_out in (1, 2, 4):
        torch.manual_seed(0)
        m = StubEncoder(768, 1000, 196)
        a = make_args("ep", 768, 1000, "vit_base_patch16")
        a.d_out = d_out
        try:
            build(m, a)
            h = hashlib.sha256()
            for n, p in sorted(m.head.named_parameters()):
                h.update(n.encode())
                h.update(p.detach().float().cpu().numpy().tobytes())
            results["ep@768/d_out=%d" % d_out] = {
                "repr": re.sub(r"\s+", " ", repr(m.head)),
                "n_params": sum(p.numel() for p in m.head.parameters()),
                "sha256": h.hexdigest(),
            }
        except Exception as e:
            results["ep@768/d_out=%d" % d_out] = {"error": "%s: %s" % (type(e).__name__, e)}

    json.dump(results, open(args.out, "w"), indent=1, sort_keys=True)
    ok = sum(1 for v in results.values() if "error" not in v)
    print("wrote %s: %d entries, %d built, %d raised" % (args.out, len(results), ok, len(results) - ok))


if __name__ == "__main__":
    main()
