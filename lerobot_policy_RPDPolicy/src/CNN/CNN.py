from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .core.features import FeatureExtraction
from .core.detect import Detector
from .core.track import Tracker
from .core.train_utils import load_image, viz
from .core.train import fit
from .core.infer import predict

class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        neck: str = "fpn",
        feature_channels: int = 256,
        bifpn_layers: int = 3,
        use_cbam: bool = False,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 10,
        image_size: tuple[int, int] = (640, 640),
        class_names: dict | list | None = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.image_size = image_size

        if class_names is None:
            self.names = {
                i: str(i)
                for i in range(num_classes)
            }

        elif isinstance(class_names, list):
            self.names = {
                i: name
                for i, name in enumerate(class_names)
            }

        else:
            self.names = class_names

        self.feature_extractor = FeatureExtraction(
            backbone_name=backbone_name,
            pretrained=pretrained,
            neck=neck,
            out_channels=feature_channels,
            bifpn_layers=bifpn_layers,
            use_cbam=use_cbam,
        )

        num_levels = self.feature_extractor.num_output_levels()

        channels = (
            feature_channels,
        ) * num_levels

        self.detector = Detector(
            num_classes=num_classes,
            channels=channels,
            strides=self.feature_extractor.out_strides,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        self.tracker = Tracker()
        self.criterion = None
        self.model_config = {
            "num_classes": num_classes,
            "backbone_name": backbone_name,
            "neck": neck,
            "feature_channels": feature_channels,
            "bifpn_layers": bifpn_layers,
            "use_cbam": use_cbam,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
            "image_size": image_size,
            "class_names": self.names,
        }

    @property
    def model(self):
        return [
            self.feature_extractor,
            self.detector.head,
        ]

    @property
    def device(self):
        return next(self.parameters()).device

    def init_criterion(self):
        self.criterion = self.detector.init_criterion()
        return self.criterion

    def forward(self, x: torch.Tensor):
        features = list(
            self.feature_extractor(x)
        )

        return self.detector(
            features
        )

    @torch.no_grad()
    def detect(
        self,
        x: torch.Tensor,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ):
        self.eval()

        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        features = list(
            self.feature_extractor(x)
        )

        return self.detector.detect(
            features,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    @torch.no_grad()
    def predict(self, source, conf_threshold: float | None = None, iou_threshold: float | None = None):
        return predict(self, source, conf_threshold, iou_threshold)

    @torch.no_grad()
    def track(
        self,
        x: torch.Tensor,
        tracker: str = "bytetrack",
        img=None,
    ):
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(
                "track() expects x with shape [1, C, H, W]."
            )

        features = list(
            self.feature_extractor(x)
        )

        detections = self.detector.detect(
            features,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

        detection = (
            detections[0]
            .detach()
            .cpu()
        )

        return self.tracker.track(
            detections=detection,
            image_shape=x.shape[-2:],
            tracker=tracker,
            img=img,
        )

    def reset_tracker(self, tracker: str | None = None):
        self.tracker.reset_tracker(tracker)

    def fit(self, dataset, epochs: int = 100, lr: float = 1e-3, optimizer=None,
            device=None, output_dir: str | Path = "./runs"):
        return fit(self, dataset, epochs, lr, optimizer, device, output_dir)