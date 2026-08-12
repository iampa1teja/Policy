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
        self.embed = nn.Embedding(max_len, embed_dim) 
        nn.init.normal_(self.embed.weight, std=0.02) 

    def forward(self, positions: torch.Tensor) -> torch.Tensor: 
        return self.embed(positions) 

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

        self.feature_encoder = TrackFeatureEncoder(embed_dim) 
        self.temporal_ensembling = TemporalEnsembling(embed_dim, max_len=max_history_len) 

        lstm_hidden_dim = lstm_hidden_dim or embed_dim 

        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=lstm_hidden_dim, 
            batch_first=True, 
        )
        self.out_proj = nn.Linear(
            lstm_hidden_dim, 
            embed_dim
        ) if lstm_hidden_dim != embed_dim else None 

        self._history: Dict[
            int,
            Deque[Tuple[float, float, float, float, int]]
        ] = {}

    def _update_history(self, tracks) -> None: 
        for t in tracks: 
            track_id = int(t.track_id) 
            box = {float(t.x), float(t.y), float(t.w), float(t.h), float(t.frame_id) }

        if track_id not in self._history: 
            self._history[track_id] = deque(maxlen=self.max_history_len) 
        self._history(track_id).append(box) 

    def reset(self, tracker = None) -> None: 
        self._history.clear() 

    def forward(self, tracks):
        self._update_history(tracks)

        device = next(self.parameters()).device
        tokens = {}

        for track_id, history in self._history.items():

            boxes = torch.tensor(
                [
                    [x, y, w, h]
                    for x, y, w, h, _ in history
                ],
                dtype=torch.float32,
                device=device,
            )

            frame_positions = torch.arange(
                len(history),
                device=device,
                dtype=torch.long,
            )

            embedded = self.feature_encoder(boxes)

            embedded = (
                embedded
                + self.temporal_ensembling(frame_positions)
            )

            _, (h_n, _) = self.lstm(
                embedded.unsqueeze(0)
            )

            token = h_n[-1, 0]

            if self.out_proj is not None:
                token = self.out_proj(token)

            tokens[track_id] = token

        return tokens