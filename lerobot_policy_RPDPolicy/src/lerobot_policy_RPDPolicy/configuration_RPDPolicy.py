from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import torch 
from ..CNN import CNN
from transformers import GemmaConfig

from lerobot.configs import PreTrainedConfig 
from lerobot.optim import AdamConfig 
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig
from lerobot.configs.types import NormalizationMode, FeatureType

class TrackerType(Enum): 
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"

class ActionNormalizationMode(str, Enum):
    IDENTITY = "identity"
    MEAN_STD = "mean_std"
    MIN_MAX = "min_max"
    QUANTILES = "quantiles"
    QUANTILE10 = "quantile10"

@PreTrainedConfig.register_subclass("RPDPolicy") 
@dataclass 
class RPDConfig(PreTrainedConfig): 
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

    action_normalization: ActionNormalizationMode = (ActionNormalizationMode.IDENTITY)

    vision_embed_dim: int = 256
    track_embed_dim: int = 256
    vision_tokens_per_level: list[int] = field(default_factory=lambda: [16, 16, 16])
    adarms_cond_dim: int = 256
    action_expert_hidden_size: int = 256

    action_expert_num_layers: int = 6
    action_expert_num_heads: int = 8
    action_expert_intermediate_size: int = 1024

    num_inference_steps: int = 10
    min_t: float = 0.0
    max_t: float = 1.0

    max_tracks: int = 16
    max_history_len: int = 30

    tokenizer_max_length: int = 48

    use_vision_tokens: bool = True
    use_track_tokens: bool = True
    freeze_perception: bool = True
    condition_weights: dict = field(
        default_factory=lambda: {"image": 1.0, "track": 1.0} 
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory= lambda: {
            "VISUAL" : NormalizationMode.IDENTITY,
            "STATE" : NormalizationMode.MEAN_STD,
            "ACTION" : NormalizationMode.MEAN_STD,
        }
    )


    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon: 
            raise ValueError("n_action_steps cannot exceed horizon") 

    def validate_features(self):
        if not self.image_features:
            raise ValueError("RPDPolicy requires at least one image feature")

        if self.action_feature is None:
            raise ValueError("RPDPolicy requires 'action' in output_features")

        if self.model_checkpoint is None:
            raise ValueError("The CNN model needs to be trained to use the policy")

        try:
            ckpt = torch.load(
                self.model_checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            required_keys = {
                "feature_extractor_state_dict",
                "detector_state_dict",
                "model_config",
            }

            missing = required_keys - ckpt.keys()

            if missing:
                raise ValueError(f"Checkpoint missing keys: {missing}")

        except Exception as e:
            raise RuntimeError(f"Unable to load CNN checkpoint: {e}") from e

    def get_optimizer_preset(self):
        return AdamConfig(lr = self.optimizer_lr, weight_decay=self.optimizer_weight_decay) 

    def get_scheduler_preset(self):
        return None 

    def make_action_expert_config(self):
        cfg = GemmaConfig(
            vocab_size=1,
            hidden_size=self.action_expert_hidden_size,
            intermediate_size = self.action_expert_intermediate_size,
            num_hidden_layers=self.action_expert_num_layers,
            num_attention_heads=self.action_expert_num_heads,
            num_key_value_heads=self.action_expert_num_heads,
            head_dim=self.action_expert_hidden_size // self.action_expert_num_heads,
            max_position_embeddings=512,
            attn_implementation="sdpa"
        )
        cfg.use_adarms = True 
        cfg.adarms_cond_dim = self.adarms_cond_dim
        return cfg
        

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

    @property
    def state_dim(self) -> int:
        return self.robot_state_feature.shape[0]
    


        