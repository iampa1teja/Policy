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

        feature_extractor = feature_extractor.load_state_dict(cktp["feature_extractor_state_dict"])
        self.feature_extractor = feature_extractor 

        detector = Detector() 
        detector = detector.load_state_dict(cktp["detect_state_dict"])
        self.detector = detector 

        self.tracker = Tracker()