import torch 
import torch.nn as nn 
from typing import Any 

from lerobot.policies import PreTrainedPolicy 
from lerobot.utils.constants import ACTION 
from.configuration_RPDPolicy import RPDPolicyConfig

from ..CNN.core.features import FeatureExtraction
from ..CNN.core.detect import Detector 
from ..CNN.core.track import Tracker

from .utils.vision_tokenizer import VisionTokenizer
from .utils.object_tokenizer import TrackTokenizer

class RPDPolicy(PreTrainedPolicy):
    config_class = RPDPolicyConfig 
    name = "RPDPolicy" 

    def __init__(self, config: RPDPolicyConfig, dataset_stats: dict[str, Any] = None): 
        super().__init__(config, dataset_stats) 
        config.validate_features() 
        self.config = config 

        cktp = torch.load(config.model_checkpoint) 

        extractor = FeatureExtraction()
        tokenizer = VisionTokenizer(
            tokens_per_level=[16, 8, 4],
            embed_dim=256,
            pos_embedding_mode="sine",
        )

        model_config = cktp["model_config"]
        feature_extractor = FeatureExtraction(
            backbone_name=model_config["backbone_name"],
            pretrained=False,
            neck=model_config["neck"],
            out_channels=model_config["feature_channels"],
            bifpn_layers=model_config["bifpn_layers"],
            use_cbam=model_config["use_cbam"],
        )

        feature_extractor.load_state_dict(
            cktp["feature_extractor_state_dict"]
        )

        self.feature_extractor = feature_extractor

        num_levels = feature_extractor.num_output_levels()
        channels = (model_config["feature_channels"],) * num_levels

        detector = Detector(
            num_classes=model_config["num_classes"],
            channels=channels,
            strides=feature_extractor.out_strides,
            conf_threshold=model_config["conf_threshold"],
            iou_threshold=model_config["iou_threshold"],
            max_detections=model_config["max_detections"],
        )

        detector.load_state_dict(
            cktp["detector_state_dict"]
        )

        self.detector = detector

        self.tracker = Tracker()

        self.track_tokenizer = TrackTokenizer(
            embed_dim = config.hidden_dim, 
            max_history_len=30, 
        )