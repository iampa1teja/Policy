from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_RPDPolicy import RPDPolicyConfig

from ..CNN.core.features import FeatureExtraction
from ..CNN.core.detect import Detector
from ..CNN.core.track import Tracker

from .utils.vision_tokenizer import VisionTokenizer, TokenProjector
from .utils.object_tokenizer import TrackTokenizer
from .utils.pi_gemma import PiGemmaModel
from .utils.flow_matching import ConditionalFlowMatching


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbedding dim must be even, got {dim}.")
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] float in [0, 1] -> [B, cond_dim]."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        sinusoid = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(sinusoid.to(t.dtype if t.is_floating_point() else torch.float32))


class ActionExpert(nn.Module):

    def __init__(
        self,
        config: RPDPolicyConfig,
        action_dim: int,
        condition_dim: int,
    ):
        super().__init__()
        gemma_config = config.make_action_expert_config()
        self.decoder = PiGemmaModel(gemma_config)
        self.hidden_size = config.action_expert_hidden_size
        self.condition_proj = (
            nn.Linear(condition_dim, self.hidden_size)
            if condition_dim != self.hidden_size
            else nn.Identity()
        )

        self.state_encoder = nn.Linear(config.state_dim, self.hidden_size)
        self.action_encoder = nn.Linear(action_dim, self.hidden_size)
        self.time_embedding = SinusoidalTimeEmbedding(
            dim=self.hidden_size, cond_dim=config.adarms_cond_dim
        )
        self.velocity_head = nn.Linear(self.hidden_size, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
            Forward pass of the action expert.
            The input sequence is constructed as:
                [prefix tokens] [state token] [action tokens]

            Attention uses a blocked prefix/action mask rather than causal
            autoregressive attention:

                conditioning → conditioning : allowed
                conditioning → actions      : blocked
                actions      → conditioning : allowed
                actions      → actions       : allowed

            This allows the entire action chunk to attend bidirectionally while
            remaining conditioned on the prefix and state.
            Args:
                noisy_actions: [B, horizon, action_dim]
                    Noisy action chunk used as the input to the flow-matching
                    action expert.
                state: [B, state_dim]
                    Robot state used as an additional conditioning token.
                t: [B]
                    Flow-matching time values.
                prefix_embeds: [B, prefix_len, condition_dim]
                    Concatenated vision/track conditioning embeddings produced by
                    the condition pipeline.
                prefix_mask: [B, prefix_len] bool
                    Validity mask for the prefix conditioning tokens.
            Returns:
                velocity: [B, horizon, action_dim]
                    Predicted action velocity for every action in the chunk.
        """

        horizon = noisy_actions.shape[1]

        prefix = self.condition_proj(prefix_embeds)
        state_embed = self.state_encoder(state).unsqueeze(1)
        action_embeds = self.action_encoder(noisy_actions)
        adarms_cond = self.time_embedding(t)

        inputs_embeds = torch.cat([prefix, state_embed, action_embeds], dim=1)
        L = inputs_embeds.shape[1]

        attention_mask = torch.ones((L, L), dtype=bool, device=inputs_embeds.device)
        cond_len = prefix_embeds.shape[1] + state_embed.shape[1] 
        attention_mask[:cond_len, cond_len:] = False
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(
            inputs_embeds.shape[0], -1, -1, -1
        )

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            adarms_cond=adarms_cond,
        )

        action_hidden = outputs.last_hidden_state[:, horizon:, :]
        return self.velocity_head(action_hidden)


class RPDPolicy(PreTrainedPolicy):
    config_class = RPDPolicyConfig
    name = "RPDPolicy"

    def __init__(self, config: RPDPolicyConfig, dataset_stats: dict[str, Any] = None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config

        ckpt = torch.load(config.model_checkpoint, map_location="cpu", weights_only=False)
        model_config = ckpt["model_config"]

        feature_extractor = FeatureExtraction(
            backbone_name=model_config["backbone_name"],
            pretrained=False,
            neck=model_config["neck"],
            out_channels=model_config["feature_channels"],
            bifpn_layers=model_config["bifpn_layers"],
            use_cbam=model_config["use_cbam"],
        )
        feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"])
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
        detector.load_state_dict(ckpt["detector_state_dict"])
        self.detector = detector

        self.tracker = Tracker()

        self.track_tokenizer = TrackTokenizer(
            embed_dim=config.track_embed_dim,
            max_history_len=config.max_history_len,
        )
        self.track_projector = TokenProjector(
            in_dim=config.track_embed_dim,
            out_dim=config.hidden_dim,
        )

        self.vision_tokenizer = VisionTokenizer(
            in_channels=channels[0],
            num_levels=num_levels,
            tokens_per_level=config.vision_tokens_per_level,
            embed_dim=config.vision_embed_dim,
            pos_embedding_mode="sine",
        )
        self.vision_projector = TokenProjector(
            in_dim=config.vision_embed_dim,
            out_dim=config.hidden_dim,
        )

        action_dim = config.action_feature.shape[0]

        self.action_expert = ActionExpert(
            config,
            action_dim=action_dim,
            condition_dim=config.hidden_dim,
        )

        self.flow_matching = ConditionalFlowMatching(
            action_expert=self.action_expert,
            hidden_size=config.hidden_dim,
            num_inference_steps=config.num_inference_steps,
            min_t=config.min_t,
            max_t=config.max_t,
        )

        self._action_queue: deque[torch.Tensor] = deque(maxlen=config.n_action_steps)


    def _image_condition(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        vision_tokens = self.vision_tokenizer(features)
        return self.vision_projector(vision_tokens)

    def _track_condition_offline(
        self,
        boxes: torch.Tensor,
        frame_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        track_tokens = self.track_tokenizer.forward_history_batch(
            boxes, frame_ids, history_mask
        )
        return self.track_projector(track_tokens), history_mask.any(dim=-1)

    def _track_condition_realtime(
        self, tracks: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.track_tokenizer(tracks)
        device = next(self.track_projector.parameters()).device
        padded = torch.zeros(
            self.config.max_tracks,
            self.config.track_embed_dim,
            device=device,
        )
        valid = torch.zeros(
            self.config.max_tracks, dtype=torch.bool, device=device
        )

        for slot, track_id in enumerate(sorted(tokens)[: self.config.max_tracks]):
            padded[slot] = tokens[track_id].to(device)
            valid[slot] = True

        return self.track_projector(padded.unsqueeze(0)), valid.unsqueeze(0)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self.track_tokenizer.reset()
        self.tracker.reset_tracker(self.config.tracker.value)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        image_key = next(iter(self.config.image_features))
        images = batch[image_key]
        state = batch[OBS_STATE]
        actions = batch[ACTION]

        image_condition = self._image_condition(images)
        track_condition, track_valid_mask = self._track_condition_offline(
            batch["track_boxes"], batch["track_frame_ids"], batch["track_history_mask"]
        )

        weights = self.config.condition_weights
        conditions = {
            "image": (weights["image"], image_condition),
            "track": (weights["track"], track_condition),
        }

        loss = self.flow_matching(
            a1=actions,
            state=state,
            conditions=conditions,
            masks={"track": track_valid_mask},
        )

        return loss, {"loss": loss.item() if torch.is_tensor(loss) else loss}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self.eval()

        image_key = next(iter(self.config.image_features))
        images = batch[image_key]
        state = batch[OBS_STATE]

        image_condition = self._image_condition(images)
        track_condition, track_valid_mask = self._track_condition_realtime(batch["tracks"])

        weights = self.config.condition_weights
        conditions = {
            "image": (weights["image"], image_condition),
            "track": (weights["track"], track_condition),
        }

        action_dim = self.config.action_feature.shape[0]

        actions = self.flow_matching(
            a1=None,
            state=state,
            conditions=conditions,
            horizon=self.config.horizon,
            action_dim=action_dim,
            masks={"track": track_valid_mask},
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()


__all__ = ["RPDPolicy", "ActionExpert", "SinusoidalTimeEmbedding"]
