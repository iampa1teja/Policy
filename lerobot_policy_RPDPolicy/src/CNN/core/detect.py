from __future__ import annotations

import torch 
import torch.nn as nn 

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.engine.results import Results 

class Detector: 
    def __init__(
        self,
        num_classes: int, 
        channels: tuple, 
        strides: list, 
        conf_threshold: float = 0.25, 
        iou_threshold: float = 0.45, 
        max_detections: int = 10, 
    ):
        super().__init__() 

        self.num_classes = num_classes

        self.conf_threshold = conf_threshold 
        self.iou_threshold = iou_threshold 
        self.max_detections = max_detections 

        self.head = Detect(
            nc = num_classes, 
            ch = channels, 
        )

        self.head_stride = torch.tensor(
            strides, 
            dtype=torch.float32, 
        )

        self.head_bias_init() 

        self.args = type(
            "Args", 
            (), 
            {
                "box": 7.5, 
                "cls": 0.5, 
                "dfl": 1.5, 
            },
        )() 
        self.criterion = None 

    def forward(
        self, 
        features, 
    ): 
        return self.head(features) 

    def init_criterion(self): 
        self.criterion = v8DetectionLoss(self) 
        return self.criterion 

    @torch.no_grad() 
    def detect(
        self, 
        features, 
        conf_threshold: float | None = None, 
        iou_threshold: float | None = None, 
    ):
        if conf_threshold is None: 
            conf_threshold = self.conf_threshold 

        if iou_threshold is None: 
            iou_threshold = self.iou_threshold 

        predictions = self.forward(features) 

        if isinstance(predictions, tuple): 
            predictions = predictions[0] 

        detections = non_max_suppression(
            prediction=predictions, 
            conf_thres=conf_threshold,
            iou_thres=iou_threshold, 
            max_det=self.max_detections, 
            nc = self.num_classes,
        )

        return detections

    def results(
        self, 
        detections, 
        image, 
        path, 
        names, 
    ):
        return Results(
            orig_img=image, 
            path=path, 
            names=names, 
            boxes=detections
        )