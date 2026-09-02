import torch
import torch.nn as nn

from typing import Optional, List
import math


class PositionEmbedding2D(nn.Module):
    def __init__(
        self,
        num_pos_feats: int,
        mode: str = "sine",
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        max_len: int = 128,
    ):
        super().__init__()
        if mode not in ("sine", "learned"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.max_len = max_len

        self.scale = scale if scale is not None else 2 * math.pi

        if self.mode == "learned":
            self.row_embed = nn.Embedding(self.max_len, num_pos_feats)
            self.col_embed = nn.Embedding(self.max_len, num_pos_feats)
            nn.init.uniform_(self.row_embed.weight)
            nn.init.uniform_(self.col_embed.weight)

    def _sine_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device = x.device

        not_mask = torch.ones(B, H, W, device=device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).float()
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, 2*num_pos_feats, H, W)
        return pos

    def _learned_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device = x.device

        i = torch.arange(W, device=device)
        j = torch.arange(H, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat(
            [
                y_emb.unsqueeze(1).expand(-1, W, -1),
                x_emb.unsqueeze(0).expand(H, -1, -1),
            ],
            dim=-1,
        )
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "sine":
            return self._sine_forward(x)
        return self._learned_forward(x)


class TokenLearner(nn.Module):
    def __init__(self, in_channels: int, num_tokens: int, hidden_channels: Optional[int] = None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.attention_maps = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_tokens, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        attn = self.attention_maps(x).flatten(2).softmax(dim=-1)
        x_flat = x.flatten(2)
        return torch.einsum("bnl,bcl->bnc", attn, x_flat)


class ScaleEmbedding(nn.Module):
    def __init__(self, num_levels: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(num_levels, embed_dim))
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, tokens: torch.Tensor, level_idx: int) -> torch.Tensor:
        return tokens + self.embeddings[level_idx].view(1, 1, -1)


class VisionTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        num_levels: int,  
        tokens_per_level: List[int],
        embed_dim: int = 256,
        pos_embedding_mode: str = "sine",
        use_positional_encoding: bool = True,
        use_scale_embedding: bool = True,
    ):
        super().__init__()
        self.tokens_per_level = tokens_per_level
        self.pos_embedding_mode = pos_embedding_mode
        self.use_positional_encoding = use_positional_encoding
        self.use_scale_embedding = use_scale_embedding

        if len(tokens_per_level) != num_levels:
            raise ValueError("MIsmatch in number of levels and the given configuration of tokens for each level.")

        if use_positional_encoding: 
            self.pos_encodings = nn.ModuleList([
                PositionEmbedding2D(num_pos_feats=in_channels // 2, mode = self.pos_embedding_mode)
                for _ in range(num_levels) 
            ])

        self.token_learners = nn.ModuleList([
            TokenLearner(in_channels, self.tokens_per_level[i])
            for i in range(num_levels)
        ])

        self.proj = nn.Linear(in_channels, embed_dim) if embed_dim != in_channels else None

        if use_scale_embedding: 
            self.scale_embedding = ScaleEmbedding(num_levels, embed_dim) 


    def _tokenize_level(self, feature_map: torch.Tensor, level_idx: int) -> torch.Tensor:
        x = feature_map
        if self.use_positional_encoding:
            x = x + self.pos_encodings[level_idx](x)
        tokens = self.token_learners[level_idx](x)
        if self.proj is not None:
            tokens = self.proj(tokens)
        if self.use_scale_embedding:
            tokens = self.scale_embedding(tokens, level_idx)
        return tokens

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        level_tokens = [self._tokenize_level(f, i) for i, f in enumerate(features)]
        return torch.cat(level_tokens, dim=1)

    @property
    def total_num_tokens(self) -> int:
        return sum(self.tokens_per_level)

class TokenProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0): 
        super().__init__() 
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim), 
            nn.Linear(in_dim, out_dim), 
            nn.GELU(), 
            nn.Dropout(dropout) 
        )

    def forward(self, tokens: torch.Tensor): 
        return self.model(tokens)