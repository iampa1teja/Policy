from __future__ import annotations 
from typing import Any 

import numpy as np 
import torch 

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, PolicyAction, TransitionKey
from lerobot.processor import (
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry, 
    make_default_policy_processor_steps,
    make_policy_processor_pipelines, 
)
from lerobot.utils.constants import OBS_PREFIX, OBS_STR

from .configuration_RPDPolicy import RPDPolicyConfig
from .utils.object_tokenizer import TrackTokenizer 
from .utils.vision_tokenizer import TokenProjector 

OBS_TRACKS = OBS_STR + ".tracks"
OBS_TRACK_EMBEDS = OBS_STR + ".track_embeds"
OBS_TRACK_VALID_MASK = OBS_STR + ".track_valid_mask"

OFFLINE_TRACK_COLUMNS = 9
RAW_TRACK_COLUMNS = 8
_TRACK_ID_COL_OFFLINE = 5
_TRACK_ID_COL_RAW = 4 

def is_offline_batch(tracks: Any) -> bool: 
    return isinstance(tracks, list) 

def _episode_to_history(
    episode: np.ndarray, 
    max_tracks: int, 
    max_history_len: int,
)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(episode, dtype=np.float32)
    boxes = np.zeros((max_tracks, max_history_len, 4), dtype=np.float32)
    frame_ids = np.zeros((max_tracks, max_history_len), dtype=np.int64) 
    mask = np.zeros((max_tracks, max_history_len), dtype=bool) 

    if rows.size == 0: 
        return boxes, frame_ids, mask 

    rows = np.atleast_2d(rows)
    if rows.shape[1] != OFFLINE_TRACK_COLUMNS:
        raise ValueError(
            f"Offline tracks must have {OFFLINE_TRACK_COLUMNS} columns "
            f"[frame_id,x1,y1,x2,y2,track_id,score,cls,idx]; got {rows.shape}."
        )

    track_ids = sorted({int(v) for v in rows[:, _TRACK_ID_COL_OFFLINE]})[-max_tracks:]

    for slot, track_id in enumerate(track_ids):
        track_rows = rows[rows[:, _TRACK_ID_COL_OFFLINE].astype(np.int64) == track_id]
        track_rows = track_rows[np.argsort(track_rows[:, 0], kind="stable")][-max_history_len:]
        length = track_rows.shape[0]
        boxes[slot, :length] = track_rows[:, 1:5]
        frame_ids[slot, :length] = track_rows[:, 0].astype(np.int64)
        mask[slot, :length] = True

    return boxes, frame_ids, mask

@ProcessorStepRegistry.register(name="rpdpolicy_track_tokenizer_processor")
class TrackTokenizerProcessorStep(ProcessorStep):
    def __init__(
        self, 
        hidden_size: int, 
        embed_dim: int = 256,
        lstm_hidden_dim: int | None = None,
        max_history_len: int = 30,
        max_tracks: int = 16, 
        device: str | torch.device = "cpu",
    ):
        self.max_tracks = max_tracks
        self.max_history_len = max_history_len
        self.embed_dim = embed_dim

        self.track_tokenizer = TrackTokenizer(
            embed_dim=embed_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            max_history_len=max_history_len
        ).to(device)

        self.track_projector = TokenProjector(
            in_dim=embed_dim,
            out_dim=hidden_size,
        ).to(device) 

    def _device(self) -> torch.device:
        return next(self.track_tokenizer.parameters()).device

    def _offline_forward(self, tracks_batch: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._device()
        bsz = len(tracks_batch)

        boxes = np.zeros((bsz, self.max_tracks, self.max_history_len, 4), dtype=np.float32)
        frame_ids = np.zeros((bsz, self.max_tracks, self.max_history_len), dtype=np.int64)
        history_mask = np.zeros((bsz, self.max_tracks, self.max_history_len), dtype=bool)

        for i, episode in enumerate(tracks_batch):
            b, f, m = _episode_to_history(episode, self.max_tracks, self.max_history_len)
            boxes[i], frame_ids[i], history_mask[i] = b, f, m

        boxes_t = torch.from_numpy(boxes).to(device)
        frame_ids_t = torch.from_numpy(frame_ids).to(device)
        mask_t = torch.from_numpy(history_mask).to(device)

        tokens = self.track_tokenizer.forward_history_batch(boxes_t, frame_ids_t, mask_t)
        track_embeds = self.track_projector(tokens)
        track_valid_mask = mask_t.any(dim=-1)
        return track_embeds, track_valid_mask

    def _realtime_forward(self, tracks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._device()
        tokens = self.track_tokenizer(tracks) 

        padded = torch.zeros(self.max_tracks, self.embed_dim, device=device)
        valid = torch.zeros(self.max_tracks, dtype=torch.bool, device=device)

        for slot, track_id in enumerate(sorted(tokens)[: self.max_tracks]):
            padded[slot] = tokens[track_id].to(device)
            valid[slot] = True

        track_embeds = self.track_projector(padded.unsqueeze(0))
        return track_embeds, valid.unsqueeze(0)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION.value) 
        if observation is None or OBS_TRACKS not in observation: 
            return transition

        tracks = observation[OBS_TRACKS]

        if is_offline_batch(tracks):
            track_embeds, track_valid_mask = self._offline_forward(tracks)
        else: 
            track_embeds, track_valid_mask = self._realtime_forward(np.asarray(tracks, dtype=np.float32))

        new_observation = dict(observation)
        new_observation[OBS_TRACK_EMBEDS] = track_embeds
        new_observation[OBS_TRACK_VALID_MASK] = track_valid_mask

        new_transition = dict(transition)
        new_transition[TransitionKey.OBSERVATION.value] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        obs_features = features.get(PipelineFeatureType.OBSERVATION, {})
        obs_features = dict(obs_features)
        obs_features[OBS_TRACK_EMBEDS] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(self.max_tracks, self.track_projector.out_dim if hasattr(self.track_projector, "out_dim") else None),
        )
        obs_features[OBS_TRACK_VALID_MASK] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(self.max_tracks,),
        )
        new_features = dict(features)
        new_features[PipelineFeatureType.OBSERVATION] = obs_features
        return new_features
    
    def reset(self) -> None:
        self.track_tokenizer.reset()

    def get_config(self) -> dict[str, Any]:
        return {
            "hidden_size": self.track_projector.model[1].out_features
            if hasattr(self.track_projector, "model")
            else None,
            "embed_dim": self.embed_dim,
            "lstm_hidden_dim": self.track_tokenizer.lstm.hidden_size,
            "max_history_len": self.max_history_len,
            "max_tracks": self.max_tracks,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for k, v in self.track_tokenizer.state_dict().items():
            state[f"track_tokenizer.{k}"] = v
        for k, v in self.track_projector.state_dict().items():
            state[f"track_projector.{k}"] = v
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        tok_state = {k[len("track_tokenizer."):]: v for k, v in state.items() if k.startswith("track_tokenizer.")}
        proj_state = {k[len("track_projector."):]: v for k, v in state.items() if k.startswith("track_projector.")}
        self.track_tokenizer.load_state_dict(tok_state)
        self.track_projector.load_state_dict(proj_state)

def make_RPDPolicy_pre_post_processors(
    config: RPDPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Pre/post-processor pipelines for RPDPolicy, mirroring
    make_pi0_pre_post_processors in lerobot/policies/pi0/processor_pi0.py.

    Pre-processing order:
        rename_observations -> add_batch_dim -> track tokenizer ->
        tokenize task text -> to_device -> normalize
    (track tokenizer runs before normalize since it produces its own
    already-projected embeddings, not a raw feature to be normalized.)

    Post-processing:
        unnormalize -> to_cpu   (reused verbatim from LeRobot; no
        hand-rolled unnormalization).
    """
    from lerobot.processor import TokenizerProcessorStep

    track_step = TrackTokenizerProcessorStep(
        hidden_size = config.hidden_dim,
        embed_dim = config.hidden_dim,
        max_history_len = getattr(config, "max_history_len", 30),
        max_tracks = getattr(config, "max_tracks", 16),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        track_step,
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
        steps.normalize,
    ]

    output_steps: list[ProcessorStep] = [steps.unnormalize, steps.to_cpu,]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)

__all__ = [
    "TrackTokenizerProcessorStep",
    "make_RPDPolicy_pre_post_processors",
    "OBS_TRACKS",
    "OBS_TRACK_EMBEDS",
    "OBS_TRACK_VALID_MASK",
]
