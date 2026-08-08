import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List
from torchvision.ops import FeaturePyramidNetwork

class FPN(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ):
        super().__init__()

        self.num_input_levels = len(in_channels_list)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(in_channels_list),
            out_channels=out_channels,
        )

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if len(features) != self.num_input_levels:
            raise ValueError(
                f"Expected {self.num_input_levels} input "
                f"feature maps, got {len(features)}"
            )
        
        x = OrderedDict(
            (str(i), feature)
            for i, feature in enumerate(features)
        )
        out = self.fpn(x)
        return list(out.values())