"""Attentive-pooling probe heads, keyed by --cls_features.

Every probe has the same shape -- a pooling module, a non-affine BatchNorm, and a
linear classifier:

    model.head = Sequential(pooling, BatchNorm1d(width, affine=False), classifier)

so the only things that vary per pooling are how the pooling module is built and,
for EP alone, how wide the classifier ends up. This file holds those differences in
a table instead of a fourteen-branch if/elif.

Two properties this must preserve exactly, because published numbers depend on them:

  * ORDER. The head is built after the seed is set, so the sequence of module
    constructions fixes the random draws. Build the pooling first, then the
    classifier -- the same order the if/elif chain used.
  * IDENTITY. `classifier` defaults to the encoder's own `model.head`, the Linear
    that the --finetune checkpoint load already wrote into. Replacing it would
    silently discard those weights, so only EP -- whose width changes -- makes a
    new one.

Any --cls_features value with no entry here (cls, gap, raw, both, ...) gets the
bare BatchNorm + classifier probe, which is plain linear probing.
"""
import torch.nn as nn

from poolings.abmilp import ABMILPHead
from poolings.aim import AttentionPoolingClassifier
from poolings.cae_att import CAEAttentiveBlock
from poolings.cbam import CbamPooling
from poolings.clip.attention_pool import AttentionPoolLatent
from poolings.clip.attention_pool2d import AttentionPool2d
from poolings.coca_pytorch import CrossAttention as CocaPooling
from poolings.dolg.dolg import SpatialAttention2d
from poolings.ep import EfficientProbing
from poolings.jepa.attentive_pooler import AttentivePooler
from poolings.other_pool import CAPooling, DinoViTBlockPooling
from poolings.simpool import SimPool, SimPool_nolinears
from util.cls_features import ATTENTIVE_POOLINGS


def _abmilp(dim, args, model):
    return ABMILPHead(
        dim=dim,
        self_attention_apply_to=args.abmilp_sa,
        activation=args.abmilp_act,
        depth=args.abmilp_depth,
        cond=args.abmilp_cond,
        content=args.abmilp_content,
        num_patches=model.patch_embed.num_patches,
    )


def _clip(dim, args, model):
    # CAPI ViT-L/14 is trained at 224/14 = 16 tokens a side; everything else here is 14.
    feat_size = 16 if args.model == "capi_vitl14_in1k" else 14
    return AttentionPool2d(in_features=dim, feat_size=feat_size)


# name -> (pooling factory, classifier factory or None to keep the encoder's own head)
#
# EP is the one pooling that replaces the classifier. It compresses the token axis
# by d_out, so the classifier it feeds is dim // d_out wide. Note it builds a fresh
# Linear even at d_out=1, where the width is unchanged -- that is deliberate and is
# what every published EP row did, so it stays.
POOLINGS = {
    "abmilp":   (_abmilp, None),
    "simpool":  (lambda dim, a, m: SimPool(dim=dim, num_heads=1, qkv_bias=False,
                                           qk_scale=None, gamma=None, use_beta=False), None),
    "esimpool": (lambda dim, a, m: SimPool_nolinears(dim=dim, num_heads=12, qk_scale=None,
                                                     gamma=None, use_beta=False), None),
    "clip":     (_clip, None),
    "siglip":   (lambda dim, a, m: AttentionPoolLatent(in_features=dim), None),
    "aim":      (lambda dim, a, m: AttentionPoolingClassifier(dim=dim, num_heads=a.num_heads), None),
    "ep":       (lambda dim, a, m: EfficientProbing(dim=dim, num_queries=a.ep_queries, d_out=a.d_out),
                 lambda dim, a: nn.Linear(dim // a.d_out, a.nb_classes, bias=True)),
    "cbam":     (lambda dim, a, m: CbamPooling(channels=dim, spatial_kernel_size=7), None),
    "coca":     (lambda dim, a, m: CocaPooling(dim=dim), None),
    "cait":     (lambda dim, a, m: CAPooling(embed_dim=dim), None),
    "dinovit":  (lambda dim, a, m: DinoViTBlockPooling(d_model=dim), None),
    "jepa":     (lambda dim, a, m: AttentivePooler(embed_dim=dim, num_heads=a.num_heads), None),
    "dolg":     (lambda dim, a, m: SpatialAttention2d(in_c=dim, s3_dim=dim, with_aspp=False), None),
    "cae":      (lambda dim, a, m: CAEAttentiveBlock(dim=dim), None),
}


def build_probe_head(model, args):
    """Replace model.head in place with the probe selected by args.cls_features.

    Must run AFTER the --finetune checkpoint load: the checkpoint's keys are
    `head.weight`/`head.bias`, and once the head is a Sequential they would be
    `head.2.weight` and would silently fail to match.
    """
    name = args.cls_features
    base = name[:-len("_all")] if name.endswith("_all") else name
    dim = model.head.in_features

    if base not in POOLINGS:
        # plain linear probe: no pooling module, the encoder decides the representation
        model.head = nn.Sequential(_batchnorm(dim), model.head)
        return

    make_pooling, make_classifier = POOLINGS[base]
    pooling = make_pooling(dim, args, model)          # built first: fixes the RNG order
    classifier = model.head if make_classifier is None else make_classifier(dim, args)
    model.head = nn.Sequential(pooling, _batchnorm(classifier.in_features), classifier)


def _batchnorm(width):
    return nn.BatchNorm1d(width, affine=False, eps=1e-6)


# The encoders remap --cls_features to a token selection using the same list of
# poolings; if the two ever drift, a pooling would get tokens it cannot pool.
assert sorted(POOLINGS) == sorted(ATTENTIVE_POOLINGS), (
    sorted(set(POOLINGS) ^ set(ATTENTIVE_POOLINGS)))
