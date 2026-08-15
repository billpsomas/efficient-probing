"""Frozen encoders, selected by --model plus the loader flags.

The benchmark draws encoders from seven upstream sources that agree on nothing --
torch.hub, timm, open_clip, diffusers, and three bespoke loaders -- so each one gets
a builder here and they are tried IN ORDER. The order is load-bearing and is the
same order the original if/elif chain used:

    model-name prefixes first (capi, dinov2, aimv2, franca, DiT/SiT, dinov3), then
    the loader flags --timm, --radio, --openclip, --simmim, then models_vit.

Two consequences worth knowing before reordering anything. `--timm` is tested before
`--simmim`, so passing both gives a timm backbone. And a prefix always wins over a
flag, so `--openclip --model dinov2_vitb14` builds a DINOv2 backbone -- while the
transforms, which key on --openclip alone, come from open_clip. The chain does not
police those combinations and neither does this file.

Every builder returns a wrapper exposing `.head` (a Linear) and, for the token-level
poolings, a `features` mode -- that uniformity is what lets probe_heads.py stay
encoder-agnostic.
"""
import torch

import models_capi
import models_more
import models_simmim
import models_vit
import util.misc as misc


def hub_load_rank0_first(repo, *args, **kwargs):
    """torch.hub.load, but let rank 0 populate the cache before the others read it.

    Concurrent first-time downloads into a shared TORCH_HOME race and corrupt the
    cache, so rank 0 loads once behind a barrier and every rank then loads for real.
    Rank 0 therefore loads twice; the second one is a cache hit and costs nothing.
    """
    if misc.is_dist_avail_and_initialized():
        if misc.get_rank() == 0:
            _ = torch.hub.load(repo, *args, **kwargs)
        torch.distributed.barrier()
    return torch.hub.load(repo, *args, **kwargs)


def _capi(args, device):
    return models_capi.CapiWrapper(
        capi_model=torch.hub.load('facebookresearch/capi:main', args.model),
        num_classes=args.nb_classes,
        features=args.cls_features,
    )


def _dinov2(args, device):
    return models_more.DinoWrapper(
        dino_model=torch.hub.load('facebookresearch/dinov2', args.model),
        num_classes=args.nb_classes,
        features=args.cls_features,
    )


def _aimv2(args, device):
    from aim.v2.utils import load_pretrained
    return models_more.AIMv2Wrapper(
        aimv2_model=load_pretrained(args.model, backend="torch"),
        num_classes=args.nb_classes, features=args.cls_features)


def _franca(args, device):
    # torch.hub defaults to the In21K weights, but release v1.0.0 of valeoai/Franca
    # never uploaded franca_vitl14_In21K.pth (and its ViT-g In21K chunk is 0 bytes),
    # so ViT-L is only obtainable with weights="LAION".
    kwargs = dict(use_rasa_head=args.use_rasa_head)
    if args.franca_weights:
        kwargs["weights"] = args.franca_weights
        # Upstream quirk: _make_franca_model defaults to img_size=224 and only
        # franca_vitb14 raises it to 518 for non-DINOv2 weights, but the ViT-L
        # LAION checkpoint is 518-trained (pos_embed is 1+37^2 = 1370 tokens).
        # Building at 224 fails the load outright; building at 518 matches the
        # checkpoint and the DINOv2-derived forward interpolates for our inputs.
        if args.franca_weights.upper() == "LAION" and args.franca_img_size:
            kwargs["img_size"] = args.franca_img_size
    return models_more.FrancaWrapper(
        franca_model=hub_load_rank0_first("valeoai/Franca", args.model, **kwargs),
        num_classes=args.nb_classes, features=args.cls_features,
        use_rasa_head=args.use_rasa_head)


def _diffusion(args, device):
    """DiT / SiT: a diffusion transformer probed through its frozen VAE latents."""
    from diffusers.models import AutoencoderKL
    if args.model.startswith("DiT"):
        from util.DiT.models import DiT_models as _MODELS
        from util.DiT.download import find_model as _find
    else:
        from util.SiT.models import SiT_models as _MODELS
        from util.SiT.download import find_model as _find
    backbone = _MODELS[args.model](input_size=args.dit_image_size // 8,
                                   num_classes=args.nb_classes)
    ckpt = args.dit_ckpt or f"{args.model.replace('/', '-')}-{args.dit_image_size}x{args.dit_image_size}.pt"
    backbone.load_state_dict(_find(ckpt))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()
    return models_more.DiTWrapper(
        dit_model=backbone, vae_model=vae, num_classes=args.nb_classes,
        features=args.cls_features, finetuning=args.finetuning)


def _dinov3(args, device):
    if not args.dinov3_weights:
        raise ValueError("--model dinov3_* requires --dinov3_weights (the weights are licence-gated)")
    return models_more.DinoWrapper(
        dino_model=hub_load_rank0_first("facebookresearch/dinov3", args.model,
                                        weights=args.dinov3_weights),
        num_classes=args.nb_classes,
        features=args.cls_features,
    )


def _timm(args, device):
    import timm
    backbone = timm.create_model(args.model, pretrained=True, num_classes=0)
    return models_more.TimmWrapper(
        timm_model=backbone,
        num_classes=args.nb_classes,
        features=args.cls_features,
        num_prefix_tokens=getattr(backbone, "num_prefix_tokens", 0),
    )


def _radio(args, device):
    return models_more.RadioWrapper(
        radio_model=hub_load_rank0_first("NVlabs/RADIO", "radio_model",
                                         version=args.model, progress=True),
        num_classes=args.nb_classes,
        features=args.cls_features,
    )


def _openclip(args, device):
    import open_clip
    backbone, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.openclip_pretrain)
    return models_more.CLIPWrapper(
        clip_model=backbone.visual,
        num_classes=args.nb_classes,
        features=args.cls_features,
    )


def _simmim(args, device):
    return models_simmim.__dict__[args.model](checkpoint_path=args.finetune)


def _models_vit(args, device):
    cls_kwargs = dict()
    if "huge" in args.model:
        cls_kwargs["class_token"] = not args.no_cls_token
    return models_vit.__dict__[args.model](num_classes=args.nb_classes, **cls_kwargs)


# (predicate, builder), tried in order -- see the module docstring.
BUILDERS = [
    (lambda a: a.model.startswith("capi"),           _capi),
    (lambda a: a.model.startswith("dinov2"),         _dinov2),
    (lambda a: a.model.startswith("aimv2"),          _aimv2),
    (lambda a: a.model.startswith("franca"),         _franca),
    (lambda a: a.model.startswith(("DiT", "SiT")),   _diffusion),
    (lambda a: a.model.startswith("dinov3"),         _dinov3),
    (lambda a: a.timm,                               _timm),
    (lambda a: a.radio,                              _radio),
    (lambda a: a.openclip,                           _openclip),
    (lambda a: a.simmim,                             _simmim),
]


def build_backbone(args, device):
    for matches, build in BUILDERS:
        if matches(args):
            return build(args, device)
    return _models_vit(args, device)
