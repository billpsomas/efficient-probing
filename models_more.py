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
        if not self.features in ["cls", "pos"]:
            # Patch tokens output embedding size is different from cls token output embedding size (cls has an extra projection layer)
            self.embed_dim = self.clip_model.trunk.num_features if self.is_timm else self.clip_model.transformer.width # check: self.clip_model.output_dim
        else:
            self.embed_dim = self.clip_model.trunk.num_features if self.is_timm else self.clip_model.output_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
    
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
            elif "all" in self.features:
                cls_tok = cls_tok.unsqueeze(1)
                all_tok = torch.concat([cls_tok, patch_tok], dim=1)
                out = self.head(all_tok)
            else:
                out = self.head(patch_tok) 
        else:
            if self.features == "cls":
                cls_tok = self.clip_model(x)
                out = self.head(cls_tok)
            elif "all" in self.features:
                cls_tok = self.clip_model(x)
                cls_tok = cls_tok.unsqueeze(1)
                patch_tok = self.clip_model.trunk.forward_features(x)
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
