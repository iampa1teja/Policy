import torch.nn as nn 
from ..feature_extraction import Backbone, FPN, BiFPN, CBAM

import warnings

class FeatureExtraction(nn.Module): 
    def __init__(
        self, 
        backbone_name: str = "resnet50", 
        pretrained: bool = True, 
        neck: str = "bifpn", 
        out_channels: int = 256, 
        bifpn_layers: int = 3, 
        use_cbam: bool = False,
    ): 
        super().__init__() 

        self.backbone = Backbone(
            name = backbone_name, 
            pretrained = pretrained 
        )

        in_channels = self.backbone.out_channels

        self.neck_type = neck.lower() 
        if neck not in ("bifpn", "fpn"): 
            warnings.warn("Recived an Invalid Input using BiFPN as the default neck.") 
            self.neck_type = "bifpn"
        
        if self.neck_type == "fpn":
            self.neck = FPN(
                in_channels_list = in_channels, 
                out_channels = out_channels
            )
        else: 
            self.neck = BiFPN(
                in_channels=in_channels,
                out_ch=out_channels,
                num_layers=bifpn_layers,
            )
        
        if use_cbam: 
            self.attention = nn.ModuleList(
                [
                    CBAM(out_channels) 
                    for _ in range(self.num_output_levels())
                ]
            )
        else: 
            self.attention = None

    @property 
    def out_strides(self): 
        return list(self.backbone.out_strides) 

    def num_output_levels(self): 
        return len(self.out_strides) 

    def forward(self, x): 
        features = self.backbone(x) 
        features = self.neck(features) 

        if self.attention: 
            features = [
                attention(feature)
                for attention, feature 
                in zip (self.attention, features) 
            ]
        return features 
        