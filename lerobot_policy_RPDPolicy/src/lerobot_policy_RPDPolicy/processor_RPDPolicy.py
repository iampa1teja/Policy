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

from .configuration_RPDPolicy import RPDConfig

OBS_TRACKS = OBS_STR + ".tracks"
OBS_TRACK_BOXES = OBS_STR + ".track_boxes"
OBS_TRACK_FRAME_IDS = OBS_STR + ".track_frame_ids"
OBS_TRACK_HISTORY_MASK = OBS_STR + ".track_history_mask"

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
)->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    tid_col = rows[:, _TRACK_ID_COL_OFFLINE].astype(np.int64)
    frame_col = rows[:, 0]

    first_seen = {}
    for tid, f in zip(tid_col, frame_col):
        tid = int(tid)
        if tid not in first_seen or f < first_seen[tid]:
            first_seen[tid] = f

    track_ids = sorted(first_seen, key=lambda t: (first_seen[t], t))[:max_tracks]

    for slot, track_id in enumerate(track_ids):
        track_rows = rows[tid_col == track_id]
        track_rows = track_rows[
            np.argsort(track_rows[:, 0], kind="stable")
        ][-max_history_len:]
        length = track_rows.shape[0]
        boxes[slot, :length] = track_rows[:, 1:5]
        frame_ids[slot, :length] = track_rows[:, 0].astype(np.int64)
        mask[slot, :length] = True

    return boxes, frame_ids, mask

@ProcessorStepRegistry.register(name="rpdpolicy_track_tokenizer_processor")
class TrackTokenizerProcessorStep(ProcessorStep):
    """
    Pure data-shaping step. It does NOT tokenize.

    Track tokenization is owned by the policy (RPDPolicy.track_tokenizer), so
    that its parameters live in policy.parameters() and receive gradients. A
    tokenizer living inside a processor step would never be optimized.

    This step reshapes raw track data into fixed-size tensors the model reads:

        offline (list of per-episode arrays):
            emits track_boxes [B, max_tracks, T, 4],
                  track_frame_ids [B, max_tracks, T],
                  track_history_mask [B, max_tracks, T]
        realtime (single frame): passes the raw tracks array through under
            OBS_TRACKS for the model's stateful realtime tokenizer to consume.
    """

    def __init__(
        self,
        max_history_len: int = 30,
        max_tracks: int = 16,
    ):
        self.max_tracks = max_tracks
        self.max_history_len = max_history_len

    def _offline_forward(
        self, tracks_batch: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = len(tracks_batch)

        boxes = np.zeros((bsz, self.max_tracks, self.max_history_len, 4), dtype=np.float32)
        frame_ids = np.zeros((bsz, self.max_tracks, self.max_history_len), dtype=np.int64)
        history_mask = np.zeros((bsz, self.max_tracks, self.max_history_len), dtype=bool)

        for i, episode in enumerate(tracks_batch):
            b, f, m = _episode_to_history(episode, self.max_tracks, self.max_history_len)
            boxes[i], frame_ids[i], history_mask[i] = b, f, m

        return (
            torch.from_numpy(boxes),
            torch.from_numpy(frame_ids),
            torch.from_numpy(history_mask),
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION.value)
        if observation is None or OBS_TRACKS not in observation:
            return transition

        tracks = observation[OBS_TRACKS]
        new_observation = dict(observation)

        if is_offline_batch(tracks):
            boxes, frame_ids, history_mask = self._offline_forward(tracks)
            new_observation[OBS_TRACK_BOXES] = boxes
            new_observation[OBS_TRACK_FRAME_IDS] = frame_ids
            new_observation[OBS_TRACK_HISTORY_MASK] = history_mask
        else:
            # Realtime: leave raw tracks in place for the model's stateful
            # tokenizer. Coerce to a float array so downstream is uniform.
            new_observation[OBS_TRACKS] = np.asarray(tracks, dtype=np.float32)

        new_transition = dict(transition)
        new_transition[TransitionKey.OBSERVATION.value] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        obs_features = features.get(PipelineFeatureType.OBSERVATION, {})
        obs_features = dict(obs_features)
        obs_features[OBS_TRACK_BOXES] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(self.max_tracks, self.max_history_len, 4),
        )
        obs_features[OBS_TRACK_FRAME_IDS] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(self.max_tracks, self.max_history_len),
        )
        obs_features[OBS_TRACK_HISTORY_MASK] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(self.max_tracks, self.max_history_len),
        )
        new_features = dict(features)
        new_features[PipelineFeatureType.OBSERVATION] = obs_features
        return new_features

    def reset(self) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {
            "max_history_len": self.max_history_len,
            "max_tracks": self.max_tracks,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

def make_RPDPolicy_pre_post_processors(
    config: RPDConfig,
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
    "OBS_TRACK_BOXES",
    "OBS_TRACK_FRAME_IDS",
    "OBS_TRACK_HISTORY_MASK",
]
