import torch
from torch import nn
import types

class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module, num_classes: int, features: str):
        super().__init__()
        self.dino_model = dino_model  # the backbone from Torch Hub
        self.embed_dim = dino_model.num_features
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.features = features

    def forward(self, x: torch.Tensor, return_backbone_features = False):
        # DINOv2 helper returns a *list* of tensors, one per requested layer.
        # We need only the last layer (n=1 → index 0).
        feat = self.dino_model.get_intermediate_layers(x, n=1, return_class_token=True)[0]

        # ── robust unpacking ────────────────────────────────────────────────────────
        if isinstance(feat, (tuple, list)):
            # newer dinov2 ⇒ last = (cls_tokens, patch_tokens)
            patch_tok, cls_tok = feat                     # shapes (B,D) / (B,N,D)
        else:
            # older dinov2 ⇒ last is a single tensor (B,1+N,D)
            cls_tok, patch_tok = feat[:, 0], feat[:, 1:]  # same shapes
        # ────────────────────────────────────────────────────────────────────────────

        if self.features == "cls":
            out = self.head(cls_tok)
        elif self.features == "pos":
            out = self.head(patch_tok.mean(dim=1))
        elif "all" in self.features:
            cls_tok = cls_tok.unsqueeze(1)
            all_tok = torch.concat([cls_tok, patch_tok], dim=1)
            out = self.head(all_tok)
        else:
            out = self.head(patch_tok) 

        if return_backbone_features:
            if self.features == "cls":
                return out, cls_tok
            else:
                return out, patch_tok
        return out

class CLIPWrapper(nn.Module):
    """
    Wraps CLIP/SigLIP vision encoder.
    """
    def __init__(self, clip_model: nn.Module, num_classes: int, features: str):
        super().__init__()
        self.clip_model = clip_model # Vision-only part
        self.is_timm = self.is_timm_model(self.clip_model)
        if not self.is_timm:
            self.clip_model.output_tokens = True
        
        self.features = features
        # Widths are measured with a dummy forward rather than read off attributes.
        # Attribute inference is not reliable across backends: EVA02-E-14 is a timm
        # trunk with timm_proj=null, so its projection to the CLIP width lives inside
        # the trunk and `trunk.num_features` reports the *unprojected* width, while
        # PE-Core pools to a width that differs from its patch width. Guessing wrong
        # only surfaces as a shape error deep in the first batch.
        cls_dim, token_dim, proj_dim = self._measure_dims()
        if self.features == "cls":
            # the pooled output carries the extra projection
            self.embed_dim = cls_dim
        elif "all" in self.features:
            # patch tokens are projected into the [CLS] space before concatenation
            if proj_dim != cls_dim:
                raise ValueError(
                    f"cannot build '{features}': projected patch tokens are {proj_dim}-d "
                    f"but the pooled [CLS] is {cls_dim}-d, so they cannot be concatenated. "
                    f"Use --cls_features ep or pos for this backbone.")
            self.embed_dim = proj_dim
        else:
            # 'pos' and the attentive poolings consume raw patch tokens
            self.embed_dim = token_dim
        self.head = nn.Linear(self.embed_dim, num_classes)

    @torch.no_grad()
    def _measure_dims(self):
        """Observe (pooled, patch, projected-patch) widths on a dummy batch."""
        size = getattr(self.clip_model, "image_size", 224)
        if isinstance(size, (tuple, list)):
            size = size[0]
        x = torch.zeros(1, 3, size, size)
        was_training = self.clip_model.training
        self.clip_model.eval()
        try:
            if not self.is_timm:
                feat = self.clip_model(x)
                if isinstance(feat, (tuple, list)):
                    cls_tok, patch_tok = feat
                else:
                    cls_tok, patch_tok = feat[:, 0], feat[:, 1:]
            else:
                patch_tok = self.clip_model.trunk.forward_features(x)
                cls_tok = self.clip_model.head(
                    self.clip_model.trunk.forward_head(patch_tok))
            try:
                proj_dim = self._project_tokens(patch_tok).shape[-1]
            except Exception:
                proj_dim = patch_tok.shape[-1]
        finally:
            self.clip_model.train(was_training)
        return cls_tok.shape[-1], patch_tok.shape[-1], proj_dim

    def _project_tokens(self, patch_tok):
        """Map patch tokens through the same projection the pooled output uses."""
        if self.is_timm:
            return self.clip_model.head(patch_tok)
        proj = getattr(self.clip_model, "proj", None)
        return patch_tok @ proj if proj is not None else patch_tok
    
    def is_timm_model(self,model):
            return 'timm' in model.__class__.__module__

    def forward(self, x: torch.Tensor, return_backbone_features = False):
        if not self.is_timm:
            feat = self.clip_model(x)
            
            if isinstance(feat, (tuple, list)):
                cls_tok, patch_tok = feat # shapes (B,D) / (B,N,D)
            else:
                # might be a single tensor (B,1+N,D)
                cls_tok, patch_tok = feat[:, 0], feat[:, 1:]  # same shapes
            
            if self.features == "cls":
                out = self.head(cls_tok)
            elif self.features == "pos":
                out = self.head(patch_tok.mean(dim=1))
            elif "all" in self.features:
                patch_tok = self._project_tokens(patch_tok)
                all_tok = torch.concat([cls_tok.unsqueeze(1), patch_tok], dim=1)
                out = self.head(all_tok)
            else:
                out = self.head(patch_tok) 
        else:
            if self.features == "cls":
                cls_tok = self.clip_model(x)
                out = self.head(cls_tok)
            elif self.features == "pos":
                patch_tok = self.clip_model.trunk.forward_features(x)
                out = self.head(patch_tok.mean(dim=1))
            elif "all" in self.features:
                # One backbone pass, not two. open_clip's TimmModel.forward is
                # head(trunk(x)) and timm's trunk(x) is forward_head(forward_features(x)),
                # so splitting it reproduces self.clip_model(x) exactly while reusing
                # the patch tokens instead of recomputing them.
                feats = self.clip_model.trunk.forward_features(x)
                cls_tok = self.clip_model.head(
                    self.clip_model.trunk.forward_head(feats)
                ).unsqueeze(1)
                patch_tok = self._project_tokens(feats)
                all_tok = torch.concat([cls_tok, patch_tok], dim=1)
                out = self.head(all_tok)
            else:
                patch_tok = self.clip_model.trunk.forward_features(x)
                out = self.head(patch_tok)
        
        if return_backbone_features:
            if self.features == "cls":
                return out, cls_tok
            else:
                return out, patch_tok
        return out


class TimmWrapper(nn.Module):
    """
    Wraps any `timm` backbone so it can be probed like the others.

    Covers encoders that have no [CLS] token and no global objective at all
    (SAM, Hiera, ConvNeXt, MIM-only EVA02), which is exactly the regime where
    linear probing understates the representation.

    `forward_features` returns different layouts across timm families, so the
    output is normalised to (B, N, C):
        (B, N, C)     plain ViT-style, kept as is
        (B, H, W, C)  Hiera / Swin style, flattened over H*W
        (B, C, H, W)  ConvNeXt / SAM neck, flattened and transposed
    """

    def __init__(self, timm_model: nn.Module, num_classes: int, features: str,
                 num_prefix_tokens: int = 0):
        super().__init__()
        self.timm_model = timm_model
        self.features = features
        self.num_prefix_tokens = num_prefix_tokens
        self.embed_dim = self._infer_dim()
        self.head = nn.Linear(self.embed_dim, num_classes)

    def _infer_dim(self):
        for attr in ("num_features", "embed_dim"):
            d = getattr(self.timm_model, attr, None)
            if isinstance(d, int):
                return d
        raise ValueError("could not infer feature dim from the timm model")

    @staticmethod
    def _to_tokens(f):
        """Normalise any timm forward_features output to (B, N, C)."""
        if f.ndim == 3:
            return f
        if f.ndim == 4:
            b, a, c, d = f.shape
            # (B, C, H, W) if dim 1 is the largest non-spatial axis, else (B, H, W, C)
            if a > d and c == d:
                return f.flatten(2).transpose(1, 2)   # (B, C, H, W)
            return f.reshape(b, a * c, d)             # (B, H, W, C)
        raise ValueError(f"unsupported feature shape {tuple(f.shape)}")

    def forward(self, x: torch.Tensor, return_backbone_features: bool = False):
        feats = self._to_tokens(self.timm_model.forward_features(x))

        n_prefix = self.num_prefix_tokens
        if n_prefix:
            cls_tok, patch_tok = feats[:, :n_prefix].mean(dim=1), feats[:, n_prefix:]
        else:
            cls_tok, patch_tok = None, feats

        if self.features == "cls":
            if cls_tok is None:
                raise ValueError(
                    "this backbone has no [CLS] token; use --cls_features pos "
                    "for global average pooling, or an attentive pooling.")
            out = self.head(cls_tok)
        elif self.features == "pos":
            out = self.head(patch_tok.mean(dim=1))
        elif "all" in self.features:
            if cls_tok is None:
                raise ValueError("this backbone has no [CLS] token; '_all' poolings need one.")
            out = self.head(torch.concat([cls_tok.unsqueeze(1), patch_tok], dim=1))
        else:
            out = self.head(patch_tok)

        if return_backbone_features:
            return out, (cls_tok if self.features == "cls" else patch_tok)
        return out


class RadioWrapper(nn.Module):
    """
    Wraps NVIDIA AM-RADIO / RADIOv2.5 (torch.hub `NVlabs/RADIO`).

    RADIO distils CLIP + DINOv2 + SAM into one backbone and returns
    (summary, spatial_features); the summary plays the role of a [CLS].
    """

    def __init__(self, radio_model: nn.Module, num_classes: int, features: str):
        super().__init__()
        self.radio_model = radio_model
        self.features = features
        # RADIO's summary and spatial features do NOT share a width (v2.5-b is
        # 2304 vs 768), so measure both with one dummy forward rather than
        # assuming a single embed_dim.
        self.summary_dim, self.spatial_dim = self._measure_dims()
        # The summary is not one wide vector needing a projection: RADIO is trained
        # with cls_token_per_teacher=True, so it is n_summary separate [CLS] tokens,
        # each already at the spatial width, concatenated along the feature axis
        # (v2.5-b: 2304 = 3 x 768, v2.5-l: 3072 = 3 x 1024). '_all' therefore just
        # reshapes them back into tokens -- no learned projection, nothing discarded.
        self.n_summary, rem = divmod(self.summary_dim, self.spatial_dim)
        self._summary_splits_cleanly = rem == 0
        self.embed_dim = self.summary_dim if features == "cls" else self.spatial_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _measure_dims(self):
        was_training = self.radio_model.training
        self.radio_model.eval()
        summary, spatial = self.radio_model(torch.zeros(1, 3, 224, 224))
        if spatial.ndim == 4:
            spatial = spatial.flatten(2).transpose(1, 2)
        self.radio_model.train(was_training)
        return int(summary.shape[-1]), int(spatial.shape[-1])

    def forward(self, x: torch.Tensor, return_backbone_features: bool = False):
        # RADIO carries its own input_conditioner (mean .4815/.4578/.4082,
        # std .2686/.2613/.2758). The dataloader has already applied ImageNet
        # normalisation, so undo it here and hand RADIO the [0,1] image it
        # expects; otherwise the input is normalised twice.
        x = x * self._in_std + self._in_mean
        summary, spatial = self.radio_model(x)
        if spatial.ndim == 4:                       # (B, C, H, W) -> (B, N, C)
            spatial = spatial.flatten(2).transpose(1, 2)

        if self.features == "cls":
            out = self.head(summary)
        elif self.features == "pos":
            out = self.head(spatial.mean(dim=1))
        elif "all" in self.features:
            if not self._summary_splits_cleanly:
                raise ValueError(
                    f"RADIO summary ({self.summary_dim}) is not a whole multiple of the "
                    f"spatial width ({self.spatial_dim}), so it cannot be split back into "
                    "[CLS] tokens; use --cls_features ep.")
            # (B, n*C) -> (B, n, C): recover the per-teacher [CLS] tokens
            summary_tokens = summary.reshape(summary.shape[0], self.n_summary, self.spatial_dim)
            out = self.head(torch.concat([summary_tokens, spatial], dim=1))
        else:
            out = self.head(spatial)

        if return_backbone_features:
            return out, (summary if self.features == "cls" else spatial)
        return out


class AIMv2Wrapper(nn.Module):
    def __init__(self, aimv2_model: nn.Module, num_classes: int, features: str):
        super().__init__()
        self.aimv2_model = aimv2_model
        self.embed_dim = aimv2_model.preprocessor.patchifier.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.features = features

    def forward(self, x: torch.Tensor, return_backbone_features = False):
        
        patch_tok = self.aimv2_model(x, output_features=False)

        if self.features == "cls":
            raise ValueError(
            "AIMv2 backbones do not use a [CLS] token. "
            "Use features='pos' for pooled features or any other value for per-patch features."
        )

        if self.features == "pos":
            gap = patch_tok.mean(dim=1)
            out = self.head(gap)
        else:
            out = self.head(patch_tok) 

        if return_backbone_features:
            if self.features == "pos":
                return out, gap
            else:
                return out, patch_tok
        return out


class FrancaWrapper(nn.Module):
    def __init__(self, franca_model: nn.Module, num_classes: int, features: str, use_rasa_head: bool = False):
        super().__init__()
        self.franca_model = franca_model  # the backbone from Torch Hub
        self.embed_dim = franca_model.num_features
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.features = features
        self.use_rasa_head = use_rasa_head

    def forward(self, x: torch.Tensor, return_backbone_features = False):
        feat = self.franca_model.forward_features(x, use_rasa_head=self.use_rasa_head)

        # ── unpacking ────────────────────────────────────────────────────────────────
        cls_tok = feat['x_norm_clstoken']
        if not self.use_rasa_head:
            patch_tok = feat['x_norm_patchtokens']
        else:
            patch_tok = feat['patch_token_rasa']
        # ────────────────────────────────────────────────────────────────────────────

        if self.features == "cls":
            out = self.head(cls_tok)
        elif self.features == "pos":
            out = self.head(patch_tok.mean(dim=1))
        elif "all" in self.features:
            cls_tok = cls_tok.unsqueeze(1)
            all_tok = torch.concat([cls_tok, patch_tok], dim=1)
            out = self.head(all_tok)
        else:
            out = self.head(patch_tok) 

        if return_backbone_features:
            if self.features == "cls":
                return out, cls_tok
            else:
                return out, patch_tok
        return out


class DiTWrapper(nn.Module):
    def __init__(self, dit_model: nn.Module, vae_model, num_classes: int, features: str, finetuning: bool = False):
        super().__init__()
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.embed_dim = dit_model.hidden_size
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.features = features
        self.finetuning = finetuning
        if finetuning:
            self.training = True
        else:
            self.training = False

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, return_backbone_features = False):
        with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = self.vae_model.encode(x).latent_dist.sample().mul_(0.18215)

        patch_tok = self.dit_model.forward_features(x, y=None, t=None, train=self.training)

        if self.features == "cls":
            raise ValueError(
            "Diffusion backbones do not use a [CLS] token. "
            "Use features='pos' for pooled features or any other value for per-patch features."
        )

        if self.features == "pos":
            gap = patch_tok.mean(dim=1)
            out = self.head(gap)
        else:
            out = self.head(patch_tok) 

        if return_backbone_features:
            if self.features == "pos":
                return out, gap
            else:
                return out, patch_tok
        return out
