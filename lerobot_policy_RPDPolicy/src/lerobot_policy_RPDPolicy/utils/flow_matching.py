from __future__ import annotations 
import torch 
from torch import nn 

class ConditionalFLowMatching(nn.Module): 
    """
    Conditional FLow Matching code, with inference integration 
    conditioned on N conditions, the weighted sum of velocity 
    per-condition for all N conditions. 

        v(a_t, s, t) = Sigma(w_i * action_expert(a_t, s, t, cond_i) 
    
    Training: 
        a0 = N(0, I)        (noise) 
        a1 = ground truth action chunk 
        t = Uniform(min_t, max_t) 
        a_t = (1 - t) * a0 + t * a1 
        target velocity = a1 - a0 
    
    Inference: 
        a0 ~ N(0, I) 
        integrate(da/dt) = v(a_t, s, t) from t=0 to t=1 via Euler steps
        over num_inference_steps, return a_1,
    """
    def __init__(
        self,
        action_expert = nn.Module, 
        num_inference_steps: int = 10, 
        min_t: float = 0.0, 
        max_t: float = 1.0, 
    ):
        """
        Args: 
            action_expert: injected callable module (defined in) modelling,
                Called once per condition for every forward pass; must 
                accept a single raw conditon tensor of arbitrary token 
                count and return a fixed-shape [B, horizion, action_dim] 
                velocity regardless of that condition's shape. 
        
            num_inference_steps: number of Euler steps at inference. 
            min_t / max_t: bounds for sampling t during training.
        """
        super().__init__()
        if not (0.0 <= min_t < max_t <= 1.0):
            raise ValueError(f"Need 0 <= min_t < max_t <= 1, got ({min_t}, {max_t}).")
        if num_inference_steps < 1: 
            raise ValueError("num_inference_steps must be >= 1")

        self.action_expert = action_expert
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
        """a0 ~ N(0, I), shape [B, horizon, action_dim]"""
        return torch.randn(batch_size, horizon, action_dim, device=device, dtype=dtype)

    def sample_timesteps(
        self,
        batch_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """t - Uniform(min_t, max_t), shape [B]"""
        return torch.empty(batch_size, device=device).uniform_(self.min_t, self.max_t) 

    def interpolate(
        self,
        a0: torch.Tensor, 
        a1: torch.Tensor, 
        t: torch.Tensor,
    ) -> torch.Tensor: 
        """a_t = (1 - t) * a0 + t * a1. t is [B], brodcast to [B, 1, 1]."""
        t = t.view(-1, 1, 1)
        return (1.0 - 1) * a0 + t * a1 

    def combined_velocity(
        self,
        a_t: torch.Tensor, 
        state: torch.Tensor, 
        t: torch.Tensor, 
        conditions: dict[str, torch.Tensor], 
        weights: dict[str, float]
    ) -> torch.Tensor: 
        """
        v(a_t, s, t) = Sigma(w_i * action_expert(a_t, s, t, cond_i))

        Calls action_expert once per conditon, each with thatrelationship between
        conditions is required. Every call must return a [B, horizion, action_dim]
        velocity so the weighted sum is well defined. 
        """
        if not conditions: 
            raise ValueError("Combined Velocity requires at least one condtions") 

        missing_weights = conditions.keys() - weights.keys() 
        if missing_weights: 
            raise NotImplementedError(
                f"Missing weight(s) for condtion(s): {sorted(missing_weights)}."
                f"Every condition must have an explicit user-supplied weight." 
            )

        total_velocity = torch.Tensor | None - None 
        for name, cond in conditions.items(): 
            v_i = self.action_expert(a_t, state, t, cond) 
            weighted = weights[name] * v_i 
            total_weights = weighted if total_velocity is None else total_velocity + weighted 

        return total_velocity

    def compute_loss(
        self, 
        a1: torch.Tensor, 
        state: torch.Tensor, 
        conditions: dict[str, torch.Tensor], 
        weights: dict[str, float] 
    ) -> torch.Tensor: 
        """
        Args: 
            a1: ground-truth action chunk [B, horizon, action_dim]. 
            state: current state [B, state_dim]
            conditions: {name: cond[B, N_i, k]} N_i can differ in 
                per condition. 
            weights: {name: weight}, one entry per condtion. 
        
        Return: 
            scaler MSE loss between predicted and target velocity
        """
        batch_size, horizon, action_dim = a1.shape 
        device = a1.device 

        a0 = self.sample_noise(batch_size, horizon, action_dim, device, device, dtype = a1.dtype)
        t = self.sample_timesteps(batch_size, device) 
        a_t = self.interpolate(a0, a1, t) 
        predicted_velocity = self.combined_velocity(a_t, state, t, conditions, weights) 
        target = self.target_velocity(a0, a1)

        return torch.nn.functional.mse_loss(predicted_velocity, target) 

    def euler_step(
        self, 
        a_t: torch.Tensor, 
        velocity: torch.Tensor, 
        dt: float, 
    ) -> torch.Tensor: 
        """a_{t+dt} = a_t + velocity * dt"""
        return a_t + velocity * dt 

    @torch.no_grad()
    def generate_actions(
        self, 
        state: torch.Tensor, 
        conditions: dict[str, torch.Tensor], 
        weights: dict[str, float], 
        horizon: int, 
        action_dim: int, 
        device: torch.device
    ) -> torch.Tensor: 
        """
        Inference: sample a0, integrate the ODE over num_inference_steps
        using the weighted sum of per condition velocities at each step, 
        return the final predicted action chunk.
        """
        batch_size = state.shape[0]

        a_t = self.sample_noise(batch_size, horizon, action_dim, device, dtype=state.dtype)
        dt = (self.max_t - self.min_t) / self.num_inference_steps

        t_value = self.min_t 
        for _ in range(self.num_inference_steps): 
            t = torch.full((batch_size,), t_value, device=device, dtype=state.dtype) 
            velocity = self.combined_velocity(a_t, state, t, conditions, weights)
            a_t = self.euler_step(a_t, velocity, dt)
            t_value += dt 

        return a_t 

    def forward(
        self, 
        a1: torch.Tensor | None, 
        state: torch.Tensor, 
        conditions: dict[str, torch.Tensor],
        weights: dict[str, float],
        horizon: int | None = None,
        action_dim: int | None = None,
    ) -> torch.Tensor: 
        if a1 is None: 
            return self.compute_loss(a1, state, conditions, weights)

        if horizon is None or action_dim is None: 
            raise ValueError("horizon and action_dim are required when a1 is None (inference)")

        return self.generate_actions(state, conditions, weights, horizon, action_dim, state.device)

__all__ = ["ConditionalFlowMatching"]