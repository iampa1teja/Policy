import torch 
import torch.nn as nn 
from typing import Any 

from lerobot.policies import PreTrainedPolicy 
from lerobot.utils.constants import ACTION 
from.configuration_RPDPolicy import RPDPolicyConfig

from ..CNN.core.features import FeatureExtraction 
from ..CNN.core.detect import Detect 
from ..CNN.core.track import Tracker 

class RPDPolicy(PreTrainedPolicy):
    config_class = RPDPolicyConfig 
    name = "RPDPolicy" 

    def __init__(self, config: RPDPolicyConfig, dataset_stats: dict[str, Any] = None): 
        super().__init__(config, dataset_stats) 
        config.validate_features() 
        self.config = config 

        self.detector = Detect() 
