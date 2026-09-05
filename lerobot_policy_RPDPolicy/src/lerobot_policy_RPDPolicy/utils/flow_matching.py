#!/usr/bin/env python
"""
Conditional Flow Matching for RPDPolicy.

Contains:
    - ConditionGate: combines N user-weighted condition streams
      (e.g. image tokens, track tokens, language tokens) into a single
      prefix embedding + attention mask, via
          result = concat( w_1 * Cond_1, w_2 * Cond_2, ..., w_n * Cond_n )
      Concatenation along the sequence dimension is used rather than an
      elementwise weighted sum, because conditions differ in token count
      (image vs. track vs. text) and only share the hidden dimension.
      Weighting is a pre-prefix ablation/control mechanism only — it does
      not alter the flow-matching velocity target computed below.

    - ConditionalFlowMatching: standard conditional flow matching
      (linear interpolation path, velocity regression) with training loss
      and inference-time Euler ODE integration. Does not own or define
      the action-expert network; `action_expert` is injected as a
      constructor argument and is expected to be a PiGemmaForCausalLM-based
      module defined in modeling_RPDPolicy.py, with the call signature:

          action_expert(
              noisy_actions: Tensor[B, horizon, action_dim],
              state: Tensor[B, state_dim],
              t: Tensor[B],
              prefix_embeds: Tensor[B, L, hidden_size],
              prefix_mask: Tensor[B, L],
          ) -> Tensor[B, horizon, action_dim]   # predicted velocity
"""

from __future__ import annotations

import torch
from torch import nn


class ConditionGate(nn.Module):
    """
    Combine N user-weighted condition token streams into one prefix.

        result = concat( w_1 * Cond_1, w_2 * Cond_2, ..., w_n * Cond_n )

    Each condition is a Tensor[B, N_i, hidden_size]. An optional parallel
    mask dict supplies per-condition validity masks (Tensor[B, N_i], bool);
    conditions without an entry are treated as fully valid. Losing mask
    information silently would break padding correctness for variable-
    length conditions (e.g. track tokens with fewer than max_tracks valid
    slots), so masks are explicit rather than assumed.
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: expected last-dim size of every condition tensor.
                Conditions with a mismatched hidden_size raise at forward
                time rather than silently misaligning.
        """
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        conditions: dict[str, tuple[float, torch.Tensor]],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            conditions: {name: (weight, embeds[B, N_i, hidden_size])}
            masks: optional {name: mask[B, N_i] bool}. Names not present
                default to an all-True mask of the matching length.

        Returns:
            gated_prefix: Tensor[B, sum(N_i), hidden_size]
            combined_mask: Tensor[B, sum(N_i)] bool
        """
        if not conditions:
            raise ValueError("ConditionGate requires at least one condition.")

        masks = masks or {}
        weighted_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []

        batch_size: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None

        for name, (weight, embeds) in conditions.items():
            if embeds.ndim != 3:
                raise ValueError(
                    f"Condition '{name}' must have shape [B, N, hidden_size], "
                    f"got {tuple(embeds.shape)}."
                )
            if embeds.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"Condition '{name}' hidden size {embeds.shape[-1]} != "
                    f"expected {self.hidden_size}."
                )

            if batch_size is None:
                batch_size = embeds.shape[0]
                device = embeds.device
                dtype = embeds.dtype
            elif embeds.shape[0] != batch_size:
                raise ValueError(
                    f"Condition '{name}' batch size {embeds.shape[0]} != "
                    f"expected {batch_size}."
                )

            weighted_chunks.append(embeds * weight)

            mask = masks.get(name)
            if mask is None:
                mask = torch.ones(
                    embeds.shape[0],
                    embeds.shape[1],
                    dtype=torch.bool,
                    device=embeds.device,
                )
            else:
                if tuple(mask.shape) != (embeds.shape[0], embeds.shape[1]):
                    raise ValueError(
                        f"Mask for '{name}' has shape {tuple(mask.shape)}, "
                        f"expected {(embeds.shape[0], embeds.shape[1])}."
                    )
                mask = mask.bool()
            mask_chunks.append(mask)

        gated_prefix = torch.cat(weighted_chunks, dim=1)
        combined_mask = torch.cat(mask_chunks, dim=1)
        return gated_prefix, combined_mask


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching training loss + inference-time ODE
    integration, conditioned on N user-weighted condition streams via
    ConditionGate.

    Training:
        a0 ~ N(0, I)                       (noise)
        a1 = ground-truth action chunk
        t ~ Uniform(min_t, max_t)
        a_t = (1 - t) * a0 + t * a1        (linear interpolation path)
        target velocity = a1 - a0
        loss = || action_expert(a_t, state, t, gated_prefix, mask)
                 - (a1 - a0) ||^2

    Inference:
        a0 ~ N(0, I)
        integrate da/dt = action_expert(...) from t=0 to t=1 via Euler
        steps over num_inference_steps, return a_1.
    """

    def __init__(
        self,
        action_expert: nn.Module,
        hidden_size: int,
        num_inference_steps: int = 10,
        min_t: float = 0.0,
        max_t: float = 1.0,
    ):
        """
        Args:
            action_expert: injected callable module (defined in
                modeling_RPDPolicy.py), not owned/redefined here.
            hidden_size: hidden dim shared by all condition tensors,
                passed through to ConditionGate.
            num_inference_steps: number of Euler steps at inference.
            min_t / max_t: bounds for sampling t during training.
        """
        super().__init__()
        if not (0.0 <= min_t < max_t <= 1.0):
            raise ValueError(f"Require 0 <= min_t < max_t <= 1, got ({min_t}, {max_t}).")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")

        self.action_expert = action_expert
        self.condition_gate = ConditionGate(hidden_size)
        self.num_inference_steps = num_inference_steps
        self.min_t = min_t
        self.max_t = max_t

    def sample_noise(
        self,
        batch_size: int,
        horizon: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """a0 ~ N(0, I), shape [B, horizon, action_dim]."""
        return torch.randn(batch_size, horizon, action_dim, device=device, dtype=dtype)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """t ~ Uniform(min_t, max_t), shape [B]."""
        return torch.empty(batch_size, device=device).uniform_(self.min_t, self.max_t)

    def interpolate(
        self,
        a0: torch.Tensor,
        a1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """a_t = (1 - t) * a0 + t * a1. t is [B], broadcast to [B, 1, 1]."""
        t = t.view(-1, 1, 1)
        return (1.0 - t) * a0 + t * a1

    def target_velocity(self, a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
        """Ground-truth velocity target: a1 - a0."""
        return a1 - a0

    def compute_loss(
        self,
        a1: torch.Tensor,
        state: torch.Tensor,
        conditions: dict[str, tuple[float, torch.Tensor]],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Full training step.

        Args:
            a1: ground-truth action chunk [B, horizon, action_dim].
            state: proprioceptive state [B, state_dim].
            conditions / masks: see ConditionGate.forward.

        Returns:
            scalar MSE loss between predicted and target velocity.
        """
        batch_size, horizon, action_dim = a1.shape
        device = a1.device

        gated_prefix, prefix_mask = self.condition_gate(conditions, masks)

        a0 = self.sample_noise(batch_size, horizon, action_dim, device, dtype=a1.dtype)
        t = self.sample_timesteps(batch_size, device)
        a_t = self.interpolate(a0, a1, t)

        predicted_velocity = self.action_expert(
            a_t, state, t, gated_prefix, prefix_mask
        )
        target = self.target_velocity(a0, a1)

        return torch.nn.functional.mse_loss(predicted_velocity, target)

    def euler_step(
        self,
        a_t: torch.Tensor,
        velocity: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """a_{t+dt} = a_t + velocity * dt."""
        return a_t + velocity * dt

    @torch.no_grad()
    def generate_actions(
        self,
        state: torch.Tensor,
        conditions: dict[str, tuple[float, torch.Tensor]],
        horizon: int,
        action_dim: int,
        device: torch.device,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Inference: gate conditions once (static across integration steps),
        sample a0, integrate the ODE over num_inference_steps, return the
        final predicted action chunk.
        """
        batch_size = state.shape[0]
        gated_prefix, prefix_mask = self.condition_gate(conditions, masks)

        a_t = self.sample_noise(batch_size, horizon, action_dim, device, dtype=state.dtype)
        dt = (self.max_t - self.min_t) / self.num_inference_steps

        t_value = self.min_t
        for _ in range(self.num_inference_steps):
            t = torch.full((batch_size,), t_value, device=device, dtype=state.dtype)
            velocity = self.action_expert(a_t, state, t, gated_prefix, prefix_mask)
            a_t = self.euler_step(a_t, velocity, dt)
            t_value += dt

        return a_t

    def forward(
        self,
        a1: torch.Tensor | None,
        state: torch.Tensor,
        conditions: dict[str, tuple[float, torch.Tensor]],
        horizon: int | None = None,
        action_dim: int | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Dispatch: a1 provided -> training loss via compute_loss.
        a1 is None -> inference via generate_actions (horizon and
        action_dim then required).
        """
        if a1 is not None:
            return self.compute_loss(a1, state, conditions, masks)

        if horizon is None or action_dim is None:
            raise ValueError("horizon and action_dim are required when a1 is None (inference).")

        return self.generate_actions(
            state, conditions, horizon, action_dim, state.device, masks
        )


__all__ = ["ConditionGate", "ConditionalFlowMatching"]
