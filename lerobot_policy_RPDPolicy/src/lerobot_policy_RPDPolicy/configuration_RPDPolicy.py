from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import torch 
from ..CNN import CNN 

from lerobot.configs import PreTrainedConfig 
from lerobot.optim import AdamConfig 
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig

class TrackerType(Enum): 
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"

@PreTrainedConfig.register_subclass("RPDPolicy") 
@dataclass 
class RPDPolicyConfig(PreTrainedConfig): 
    """
    Configuration class for RPDPolicy 

    Args: 
        n_obs_steps: Number of observation steps to use as input. 
        horizion: Action prediction horizon.
        n_action_steps: Number of action steps to execute.
        hidden_dim: Hidded dimension for the policy network.
        model_checkpoint: The checkpoint of CNN tracker's weights.
    """
    horizon: int = 50 
    n_action_steps: int = 50 
    hidden_dim: int = 256 

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    model_checkpoint: Path | None = None
    tracker: TrackerType = TrackerType.BYTETRACK

    def __post_init__(self):
        return super().__post_init__()
        if self.n_action_steps > self.horizon: 
            raise ValueError("n_action_steps cannot exceed horizon") 

    def validate_features(self):
        """
        Validate input/output feature compatability. 
        """
        if not self.image_features: 
            raise ValueError("RPDPOlicy requires at least one image feature")

        if self.action_feature is None: 
            raise ValueError("RPDPolicy requreis 'action' in ouptut_features")

        if not self.model_checkpoint: 
            raise ValueError("The CNN model needs to be trained to use the policy") 
        else: 
            try: 
                ckpt = torch.load(self.model_checkpoint)
                model = CNN() 
                model = model.load_state_dict(ckpt["model_state_dict"])
            except Exception as E: 
                raise RuntimeError(f"UNable to load model weights exited with exception: {E}")

    def get_optimizer_preset(self):
        return AdamConfig(lr = self.optimizer_lr, weight_decay=self.optimizer_weight_decay) 

    def get_scheduler_preset(self):
        return None 

    @property 
    def observation_delta_indices(self) -> list[int] | None: 
        """
            Relative timestep offsets the dataset loader provides per observationn. 
            
            Return "None" for single-frame policies. For temporal policies that consume 
            multiple past or future frames, returns a list of offsets, e.g, '[-20, -10, 0, 10]' for 
            3 past frames at stride 10 and 1 future frame at stride 10. 
        """
        return None 

    @property 
    def action_delta_indices(self) -> list[int]: 
        """Relative timestep offsets for the action chunk the dataset loader returns"""
        return list(range(self.horizon)) 

    @property 
    def reward_delta_indices(self) -> None: 
        return None 



        