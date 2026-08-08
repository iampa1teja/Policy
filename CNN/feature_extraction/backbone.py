import timm
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        in_strides=(8, 16, 32),
    ):
        super().__init__()

        self.net = timm.create_model(
            name,
            features_only=True,
            pretrained=pretrained,
        )

        feature_info = self.net.feature_info

        if hasattr(feature_info, "channels"):
            all_channels = feature_info.channels()
            all_strides = feature_info.reduction()
        else:
            all_channels = [
                info["num_chs"]
                for info in feature_info
            ]
            all_strides = [
                info["reduction"]
                for info in feature_info
            ]

        self.level_idx = [
            i
            for i, stride in enumerate(all_strides)
            if stride in in_strides
        ]

        if len(self.level_idx) != len(in_strides):
            raise ValueError(
                f"Backbone '{name}' does not provide "
                f"requested strides {in_strides}. "
                f"Available strides: {all_strides}"
            )

        self.out_channels = [
            all_channels[i]
            for i in self.level_idx
        ]

        self.out_strides = [
            all_strides[i]
            for i in self.level_idx
        ]

    def forward(self, x):
        features = self.net(x)

        return [
            features[i]
            for i in self.level_idx
        ]