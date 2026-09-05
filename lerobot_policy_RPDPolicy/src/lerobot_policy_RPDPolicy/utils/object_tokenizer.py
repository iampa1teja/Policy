from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TrackFeatureEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.model(boxes)


class AgeEmbedding(nn.Module):
    """Learned embedding of age-from-newest (0 = most recent observation)."""

    def __init__(self, embed_dim: int, max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.max_len = max_len

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() and (
            positions.min() < 0 or positions.max() >= self.max_len
        ):
            raise ValueError(
                f"Temporal positions must be in [0, {self.max_len - 1}], "
                f"got range [{int(positions.min())}, {int(positions.max())}]."
            )
        return self.embed(positions)


class TrackTokenizer(nn.Module):
    """
    Converts track histories into learned track tokens.

    Realtime path:
        forward(tracks) -> Dict[int, Tensor[embed_dim]]

    Training path:
        forward_history_batch(boxes, frame_ids, history_mask)
            -> Tensor[B, N_tracks, embed_dim]

    The batched training path avoids replaying every frame through the LSTM.
    Instead, the complete per-track history window is prepared once by the
    processor and all valid track histories are processed by one packed LSTM call.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        lstm_hidden_dim: Optional[int] = None,
        max_history_len: int = 30,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_history_len = max_history_len

        self.feature_encoder = TrackFeatureEncoder(embed_dim)
        self.age_embedding = AgeEmbedding(
            embed_dim,
            max_len=max_history_len,
        )

        lstm_hidden_dim = lstm_hidden_dim or embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        self.out_proj = (
            nn.Linear(lstm_hidden_dim, embed_dim)
            if lstm_hidden_dim != embed_dim
            else None
        )

        self._history: dict[int, list[tuple[float, float, float, float, int]]] = {}
        self._frame_counter = 0

    def _update_history(self, tracks) -> None:
        """Update realtime history from raw Tracker.track() rows."""
        for track in tracks:
            if len(track) < 8:
                raise ValueError(
                    f"Tracker row must contain at least 8 columns, got {len(track)}."
                )

            x1, y1, x2, y2 = track[:4]
            track_id = int(track[4])

            box = (
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                self._frame_counter,
            )

            history = self._history.setdefault(track_id, [])
            history.append(box)
            if len(history) > self.max_history_len:
                del history[: len(history) - self.max_history_len]

        self._frame_counter += 1

    def reset(self, tracker=None) -> None:
        self._history.clear()
        self._frame_counter = 0

    def forward(self, tracks) -> dict[int, torch.Tensor]:
        """Realtime one-frame update; preserves state across frames."""
        self._update_history(tracks)

        if not self._history:
            return {}

        device = next(self.parameters()).device
        track_ids = list(self._history.keys())
        boxes, frame_ids, lengths = [], [], []

        for track_id in track_ids:
            history = self._history[track_id]
            boxes.append([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in history])
            frame_ids.append([frame_id for *_, frame_id in history])
            lengths.append(len(history))

        max_len = max(lengths)
        batch_boxes = torch.zeros(
            len(track_ids), max_len, 4, dtype=torch.float32, device=device
        )
        batch_frames = torch.zeros(
            len(track_ids), max_len, dtype=torch.long, device=device
        )
        mask = torch.zeros(
            len(track_ids), max_len, dtype=torch.bool, device=device
        )

        for i, (track_boxes, track_frame_ids) in enumerate(zip(boxes, frame_ids)):
            length = len(track_boxes)
            batch_boxes[i, :length] = torch.tensor(track_boxes, device=device)
            batch_frames[i, :length] = torch.tensor(track_frame_ids, device=device)
            mask[i, :length] = True

        tokens = self._encode_histories(batch_boxes, batch_frames, mask)
        return {track_id: tokens[i] for i, track_id in enumerate(track_ids)}

    def forward_history_batch(
        self,
        boxes: torch.Tensor,
        frame_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched training path.

        Args:
            boxes: [B, N, T, 4]
            frame_ids: [B, N, T]
            history_mask: [B, N, T] bool

        Returns:
            [B, N, embed_dim]
        """
        if boxes.ndim != 4 or boxes.shape[-1] != 4:
            raise ValueError(f"boxes must be [B,N,T,4], got {tuple(boxes.shape)}")
        if frame_ids.shape != boxes.shape[:3]:
            raise ValueError(
                f"frame_ids must be [B,N,T], got {tuple(frame_ids.shape)}"
            )
        if history_mask.shape != boxes.shape[:3]:
            raise ValueError(
                f"history_mask must be [B,N,T], got {tuple(history_mask.shape)}"
            )

        bsz, n_tracks, seq_len, _ = boxes.shape
        flat_boxes = boxes.reshape(bsz * n_tracks, seq_len, 4)
        flat_frames = frame_ids.reshape(bsz * n_tracks, seq_len)
        flat_mask = history_mask.reshape(bsz * n_tracks, seq_len).bool()

        lengths = flat_mask.sum(dim=1)
        valid_tracks = lengths > 0

        output = torch.zeros(
            bsz * n_tracks,
            self.embed_dim,
            device=boxes.device,
            dtype=boxes.dtype,
        )

        if not valid_tracks.any():
            return output.view(bsz, n_tracks, self.embed_dim)

        valid_boxes = flat_boxes[valid_tracks]
        valid_frames = flat_frames[valid_tracks]
        valid_mask = flat_mask[valid_tracks]
        valid_lengths = lengths[valid_tracks]

        # Histories are right-padded. Convert absolute frame ids to age-from-current:
        # newest valid frame has age 0, older observations have larger ages.
        last_positions = valid_lengths - 1
        last_frame_ids = valid_frames.gather(
            1,
            last_positions.unsqueeze(1),
        ).squeeze(1)

        ages = last_frame_ids.unsqueeze(1) - valid_frames
        ages = ages.clamp(min=0, max=self.max_history_len - 1).long()
        ages = ages.masked_fill(~valid_mask, 0)

        valid_embeds = self.feature_encoder(valid_boxes)
        valid_embeds = valid_embeds + self.age_embedding(ages)

        packed = pack_padded_sequence(
            valid_embeds,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        valid_tokens = h_n[-1]

        if self.out_proj is not None:
            valid_tokens = self.out_proj(valid_tokens)

        output[valid_tracks] = valid_tokens
        return output.view(bsz, n_tracks, self.embed_dim)

    def _encode_histories(
        self,
        boxes: torch.Tensor,
        frame_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Internal alias used by realtime; expects [N,T,4]."""
        return self.forward_history_batch(
            boxes.unsqueeze(0),
            frame_ids.unsqueeze(0),
            history_mask.unsqueeze(0),
        ).squeeze(0)