import torch
from torch import nn

class CapiWrapper(nn.Module):
    """
    A thin wrapper around the CAPI model from Facebook:
    - We load the model from torch.hub.
    - We define a `head` attribute for linear probing.
    - The forward returns global_repr -> we pass it to self.head.
    """
    def __init__(self, capi_model: nn.Module, num_classes: int, features: str, embed_dim: int = 1024):
        super().__init__()
        self.capi_model = capi_model  # the backbone from Torch Hub
        # By default, let's define a simple linear head (like an nn.Linear(embed_dim, num_classes)).
        # Or set it to nn.Identity if you plan to override externally.
        self.head = nn.Linear(embed_dim, num_classes)
        self.features = features

    def forward(self, x: torch.Tensor, return_backbone_features = False):
        # The CAPI model typically returns (global_repr, registers, feature_map).
        global_repr, registers, feature_map = self.capi_model(x)
        # Then pass global_repr to the linear head
        if self.features == 'cls':
            out = self.head(global_repr)
        else:
            feature_map = feature_map.view(feature_map.size(0), -1, feature_map.size(-1))
            out = self.head(feature_map)
        if return_backbone_features:
            if self.features == 'cls':
                return out, global_repr
            else:
                return out, feature_map
        return out