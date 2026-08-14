"""Dump frozen-backbone patch tokens for a handful of validation images.

Split out from the plotting so the expensive part (loading a multi-billion-parameter
encoder) happens once per model, on a GPU node, while the figure can be re-drawn
cheaply afterwards with different cluster counts.

Only the loaders the visualisation actually needs are wired up here; anything else
exits with a clear message rather than silently producing the wrong tokens.
"""
import argparse, os, sys, glob
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VAL = "/scratch/project_465003083/psomasva/datasets/imagenet/val"
BIRD_WNIDS = ["n01530575", "n01531178", "n01560419", "n01580077",
              "n01592084", "n01614925", "n01824575"]


def pick_images(n, size):
    """n images spread evenly over the bird classes, deterministic (sorted order).

    Spreading rather than taking one class keeps both questions answerable: whether
    a query group tracks the same part within a species, and whether it survives
    across species.
    """
    tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(size), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    per = max(1, -(-n // len(BIRD_WNIDS)))          # ceil
    out, names = [], []
    for w in BIRD_WNIDS:
        for f in sorted(glob.glob(os.path.join(VAL, w, "*")))[:per]:
            if len(out) >= n:
                break
            out.append(tf(Image.open(f).convert("RGB")))
            names.append("%s/%s" % (w, os.path.basename(f).rsplit(".", 1)[0][-8:]))
    return torch.stack(out), names


@torch.no_grad()
def tokens_for(model_name, loader, ft, device, x):
    """Return (B, N, C) patch tokens, [CLS] excluded."""
    if loader == "timm":
        import timm
        m = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
        f = m.forward_features(x)
        if f.ndim == 4:                      # (B,H,W,C) or (B,C,H,W)
            b, a, c, d = f.shape
            f = f.flatten(2).transpose(1, 2) if (a > d and c == d) else f.reshape(b, a * c, d)
        return f[:, m.num_prefix_tokens:] if getattr(m, "num_prefix_tokens", 0) else f
    if loader == "openclip":
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=ft)
        v = m.visual.eval().to(device)
        v.output_tokens = True
        out = v(x)
        return out[1] if isinstance(out, (tuple, list)) else out[:, 1:]
    if loader == "hub_dinov2":
        m = torch.hub.load("facebookresearch/dinov2", model_name).eval().to(device)
        return m.forward_features(x)["x_norm_patchtokens"]
    if loader == "vit":                       # models_vit + --finetune checkpoint
        import models_vit
        m = models_vit.__dict__[model_name](num_classes=0).eval().to(device)
        sd = torch.load(ft, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "decoder", "l_decoder"))}
        # drop the NaN final LayerNorm some converted checkpoints carry
        sd = {k: v for k, v in sd.items()
              if not (isinstance(v, torch.Tensor) and torch.isnan(v).any())}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print("  loaded backbone (missing=%d unexpected=%d)" % (len(missing), len(unexpected)))
        # "pos" is models_vit's name for the patch tokens with the [CLS] slice removed
        toks, _, _ = m.forward_features(x, return_features="pos")
        return toks
    raise SystemExit("unsupported loader: %s" % loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--loader", required=True,
                    choices=["timm", "openclip", "hub_dinov2", "vit"])
    ap.add_argument("--finetune", default=None, help="openclip pretrained tag")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, names = pick_images(a.n, a.input_size)
    t = tokens_for(a.model, a.loader, a.finetune, dev, x.to(dev))
    t = t.float().cpu().numpy()
    print("  tokens:", t.shape)
    np.savez_compressed(a.out, tokens=t, images=x.numpy(), names=np.array(names))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
