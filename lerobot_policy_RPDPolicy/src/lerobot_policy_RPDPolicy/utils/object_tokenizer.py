import torch
import torch.nn as nn

from collections import deque
from typing import Dict, Deque, Tuple, Optional


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


class TemporalEnsembling(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 128):
        super().__init__()

        self.max_len = max_len
        self.embed = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, frame_ids: torch.Tensor) -> torch.Tensor:
        if frame_ids.numel() == 0:
            return frame_ids.new_empty((0, self.embed.embedding_dim))

        relative_age = frame_ids[-1] - frame_ids
        relative_age = relative_age.clamp(
            min=0,
            max=self.max_len - 1,
        )

        return self.embed(relative_age)


class TrackTokenizer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        lstm_hidden_dim: Optional[int] = None,
        max_history_len: int = 30,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_history_len = max_history_len

        self.feature_encoder = TrackFeatureEncoder(
            embed_dim=embed_dim,
        )

        self.temporal_ensembling = TemporalEnsembling(
            embed_dim=embed_dim,
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

        self._history: Dict[
            int,
            Deque[Tuple[float, float, float, float, int]],
        ] = {}

        self._frame_counter = 0

    def _update_history(self, tracks) -> None:
        for track in tracks:
            x1, y1, x2, y2 = track[:4]
            track_id = int(track[4])

            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)

            box = (
                x,
                y,
                w,
                h,
                self._frame_counter,
            )

            if track_id not in self._history:
                self._history[track_id] = deque(
                    maxlen=self.max_history_len
                )

            self._history[track_id].append(box)

        self._frame_counter += 1

    def reset(self, tracker=None) -> None:
        self._history.clear()
        self._frame_counter = 0

    def forward(self, tracks) -> Dict[int, torch.Tensor]:
        self._update_history(tracks)

        device = next(self.parameters()).device
        tokens: Dict[int, torch.Tensor] = {}

        for track_id, history in self._history.items():
            boxes = torch.tensor(
                [
                    [x, y, w, h]
                    for x, y, w, h, _ in history
                ],
                dtype=torch.float32,
                device=device,
            )

            frame_ids = torch.tensor(
                [
                    frame_id
                    for _, _, _, _, frame_id in history
                ],
                dtype=torch.long,
                device=device,
            )

            bbox_embedding = self.feature_encoder(boxes)
            temporal_embedding = self.temporal_ensembling(frame_ids)

            lstm_input = (
                bbox_embedding + temporal_embedding
            )

            _, (h_n, _) = self.lstm(
                lstm_input.unsqueeze(0)
            )

            token = h_n[-1, 0]

            if self.out_proj is not None:
                token = self.out_proj(token)

            tokens[track_id] = token

        return tokens
