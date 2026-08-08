import torch.nn as nn 
from .feature_extraction import Backbone, FPN, BiFPN, CBAM

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

        neck = neck.lower() 
        if neck not in ("bifpn", "fpn"): 
            raise V