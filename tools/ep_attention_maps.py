"""Render the complementary-attention figure for a trained EP head.

The README's "Emerging Properties" figure uses EP_8, where eight queries can each
get their own colour. Our benchmark runs at Q=32, which is far too many colours to
read, so the maps are merged first: the 32 query maps are clustered by what they
attend to, and each cluster is drawn in one colour. If EP's queries really do
specialise into object parts, the clusters should land on parts rather than
scattering, and the picture should stay legible.

Nothing here touches the training path. The attention is recomputed from the saved
head exactly as poolings/ep.py does it:

    q    = cls_token * scale                       (Q, C)
    attn = softmax(q @ x^T, dim=-1)                (Q, N)

so the figure is a function of the published checkpoint and nothing else.
"""
import argparse, os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# A few bird classes -- the README figure uses birds because the parts (beak, tail,
# wing, feet) are visually distinct, which makes query specialisation easy to judge.
BIRD_WNIDS = ["n01530575", "n01531178", "n01560419", "n01580077",
              "n01592084", "n01614925", "n01824575"]


def load_head(ckpt_path):
    """Return (cls_token, num_queries, dim) from a saved EP head."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    key = next((k for k in sd if k.endswith("cls_token")), None)
    if key is None:
        raise SystemExit(f"{ckpt_path}: no cls_token in the checkpoint -- not an EP head?\n"
                         f"  keys: {list(sd)[:8]}")
    t = sd[key]                                   # (1, Q, C)
    return t[0].float(), t.shape[1], t.shape[2]


@torch.no_grad()
def ep_attention(tokens, cls_token):
    """attn[q, n]: how much query q attends to patch n. Mirrors poolings/ep.py."""
    C = tokens.shape[-1]
    scale = (C ** -0.5)
    q = cls_token * scale                          # (Q, C)
    attn = q @ tokens.transpose(0, 1)              # (Q, N)
    return attn.softmax(dim=-1)


def cluster_queries(attn, k):
    """Group the Q attention maps into k clusters by cosine similarity of the maps.

    Queries that attend to the same region get merged, which is what makes Q=32
    readable: we are asking "how many distinct things do the queries look at?",
    not "what does query 17 do?".

    `attn` may be a single image's (Q, N) map or several images concatenated along
    the patch axis into (Q, n*N). The concatenated form is what we actually use:
    clustering per image would give each panel its own arbitrary colour order and
    destroy the very thing the figure is meant to show -- that a given query group
    lands on the same part from one image to the next. One assignment, all images.
    """
    A = attn / (attn.norm(dim=1, keepdim=True) + 1e-8)
    Q = A.shape[0]
    k = min(k, Q)
    # k-means++ style seeding, then a few Lloyd iterations. Deterministic: the first
    # centre is the highest-entropy query rather than a random pick.
    ent = -(attn * (attn + 1e-12).log()).sum(1)
    centres = [A[int(ent.argmax())]]
    for _ in range(k - 1):
        d = 1 - torch.stack([A @ c for c in centres], 0).max(0).values
        centres.append(A[int(d.argmax())])
    C = torch.stack(centres, 0)
    for _ in range(25):
        assign = (A @ C.transpose(0, 1)).argmax(1)
        for j in range(k):
            m = assign == j
            if m.any():
                v = A[m].mean(0)
                C[j] = v / (v.norm() + 1e-8)
    return (A @ C.transpose(0, 1)).argmax(1)


# Fully saturated and far apart in hue: tab10 mixes in several muted tones that are
# hard to tell apart once they are modulated by attention strength.
PALETTE = np.array([
    [1.00, 0.10, 0.10],   # red
    [0.10, 0.45, 1.00],   # blue
    [0.10, 0.90, 0.20],   # green
    [1.00, 0.65, 0.00],   # orange
    [0.85, 0.10, 1.00],   # magenta
    [0.00, 0.90, 0.95],   # cyan
    [1.00, 0.95, 0.10],   # yellow
    [0.55, 0.30, 0.95],   # violet
    [1.00, 0.45, 0.70],   # pink
    [0.55, 0.85, 0.10],   # lime
])


def colourise(attn, assign, k, grid, out_hw, gamma=0.55):
    """One colour per cluster; a pixel takes the colour of the cluster attending most.

    `gamma` < 1 lifts the mid-range of the attention strength so a query group covers
    a visible area instead of a few near-white pixels at its peak.
    """
    # Mean, not sum: a cluster holding 20 queries would otherwise carry 20x the mass
    # of a single-query cluster and win the argmax over the whole image.
    merged = torch.stack([attn[assign == j].mean(0) if (assign == j).any()
                          else torch.zeros(attn.shape[1]) for j in range(k)], 0)
    gh, gw = grid
    merged = merged.reshape(k, gh, gw)[None]
    merged = F.interpolate(merged, size=out_hw, mode="bilinear", align_corners=False)[0]
    top = merged.argmax(0).numpy()
    # Opacity from the absolute attention level, scaled by a high percentile rather
    # than min-max: patches nothing attends to must stay grey, otherwise the argmax
    # paints the background with whichever group happens to win on noise.
    strength = merged.max(0).values
    # Attention is a softmax over patches, so every patch carries some mass and a
    # plain min-max leaves the background visible. Anchor the floor at a mid
    # percentile instead: anything a group does not attend to more than typically
    # goes fully grey, and only genuine peaks take colour.
    flat = strength.flatten().float()
    lo = torch.quantile(flat, 0.65)
    hi = torch.quantile(flat, 0.99)
    strength = ((strength - lo) / (hi - lo + 1e-8)).clamp(0, 1) ** gamma
    rgb = PALETTE[top % len(PALETTE)]
    return rgb, strength.numpy(), merged


def overlay(img, rgb, strength):
    """Saturated colour over a grey photo, so hue carries the query group and nothing else."""
    grey = (img * np.array([0.299, 0.587, 0.114])).sum(-1)
    grey = np.stack([grey] * 3, -1) * 0.55 + 0.10          # dim, so colour dominates
    a = strength[..., None]                                # 0 where nothing attends
    return np.clip(grey * (1 - a) + rgb * a, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="an outputs/.../linprobe_*_ep_* directory")
    ap.add_argument("--tokens", required=True, help=".npz of backbone tokens from dump_tokens")
    ap.add_argument("--clusters", type=int, default=8, help="colours to merge Q queries into")
    ap.add_argument("--gamma", type=float, default=0.55,
                    help="<1 spreads attention strength so groups cover a visible area")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cls_token, Q, C = load_head(os.path.join(args.run_dir, "checkpoint-.pth"))
    d = np.load(args.tokens)
    tokens, imgs, names = d["tokens"], d["images"], d["names"]
    if tokens.shape[-1] != C:
        raise SystemExit(f"token width {tokens.shape[-1]} != head width {C}")

    n = len(tokens)
    # One clustering over every image, so a colour means the same query group in all
    # of them. With 35 images this is the whole point: the figure is only evidence of
    # "stable semantic correspondence" if the mapping is fixed across the set.
    attns = [ep_attention(torch.from_numpy(tokens[i]).float(), cls_token) for i in range(n)]
    assign = cluster_queries(torch.cat(attns, dim=1), args.clusters)
    ngroups = len(set(assign.tolist()))
    print("  %d queries -> %d groups, shared across all %d images" % (Q, ngroups, n))

    run = os.path.basename(args.run_dir.rstrip("/"))
    per_dir = args.out or os.path.join(args.run_dir, "ep_attention")
    os.makedirs(per_dir, exist_ok=True)

    for i in range(n):
        x = torch.from_numpy(tokens[i]).float()
        gh = gw = int(round(x.shape[0] ** 0.5))
        img = np.clip(imgs[i].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        rgb, strength, _ = colourise(attns[i], assign, args.clusters, (gh, gw), img.shape[:2],
                                     gamma=args.gamma)
        f, ax = plt.subplots(1, 2, figsize=(6.4, 3.4))
        ax[0].imshow(img); ax[0].set_title(str(names[i]), fontsize=8)
        ax[1].imshow(overlay(img, rgb, strength))
        ax[1].set_title("EP Q=%d -> %d query groups" % (Q, ngroups), fontsize=8)
        for a2 in ax:
            a2.set_xticks([]); a2.set_yticks([])
        f.suptitle(run, fontsize=9)
        f.tight_layout()
        f.savefig(os.path.join(per_dir, "%03d_%s.png" % (i, str(names[i]).replace("/", "_"))),
                  dpi=120, bbox_inches="tight")
        plt.close(f)

    # plus one contact sheet so the whole set can be judged at a glance
    cols = 8
    rowsn = -(-n // cols)
    f, axes = plt.subplots(rowsn, cols, figsize=(2.0 * cols, 2.1 * rowsn))
    axes = np.atleast_2d(axes)
    for i in range(rowsn * cols):
        a2 = axes[i // cols, i % cols]
        a2.set_xticks([]); a2.set_yticks([])
        if i >= n:
            a2.axis("off"); continue
        x = torch.from_numpy(tokens[i]).float()
        gh = gw = int(round(x.shape[0] ** 0.5))
        img = np.clip(imgs[i].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        rgb, strength, _ = colourise(attns[i], assign, args.clusters, (gh, gw), img.shape[:2],
                                     gamma=args.gamma)
        a2.imshow(overlay(img, rgb, strength))
    f.suptitle("%s  --  EP Q=%d merged into %d query groups, shared colours across %d images"
               % (run, Q, ngroups, n), fontsize=11)
    f.tight_layout()
    sheet = os.path.join(args.run_dir, "ep_attention_maps.png")
    f.savefig(sheet, dpi=130, bbox_inches="tight")
    plt.close(f)
    print("wrote %d PNGs to %s" % (n, per_dir))
    print("wrote", sheet)


if __name__ == "__main__":
    main()
