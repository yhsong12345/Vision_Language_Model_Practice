"""
Diffusion Action Head for VLA Models

Diffusion-based action prediction for:
- Multi-modal action distributions
- Complex action spaces
- High-quality trajectory generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionActionHead(nn.Module):
    """
    Diffusion-based Action Prediction Head.

    Uses DDPM (Denoising Diffusion Probabilistic Models) for action generation.
    Supports:
    - Conditional generation on vision-language features
    - Action chunking
    - Multi-modal action distributions

    Args:
        input_dim: Conditioning feature dimension
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers in denoiser
        chunk_size: Number of actions to predict
        num_diffusion_steps: Number of diffusion steps
        beta_start: Starting noise schedule value
        beta_end: Ending noise schedule value
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        chunk_size: int = 10,
        num_diffusion_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_diffusion_steps = num_diffusion_steps

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition projection
        self.cond_proj = nn.Linear(input_dim, hidden_dim)

        # Noise prediction network
        noise_input_dim = action_dim * chunk_size + hidden_dim * 2  # action + time + condition

        layers = []
        dims = [noise_input_dim] + [hidden_dim] * num_layers + [action_dim * chunk_size]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())

        self.noise_pred = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.

        Args:
            features: (batch, input_dim) conditioning features
            actions: (batch, chunk_size, action_dim) ground truth for training

        Returns:
            Dict with predicted_actions and optional loss
        """
        if self.training and actions is not None:
            return self._training_forward(features, actions)
        else:
            return self._inference_forward(features)

    def _training_forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with diffusion loss."""
        B = features.shape[0]
        device = features.device

        # Flatten actions
        actions_flat = actions.view(B, -1)

        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(actions_flat)

        # Get noisy actions
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        noisy_actions = sqrt_alpha * actions_flat + sqrt_one_minus_alpha * noise

        # Predict noise
        pred_noise = self._predict_noise(noisy_actions, t, features)

        # MSE loss on noise prediction
        loss = F.mse_loss(pred_noise, noise)

        return {
            "loss": loss,
            "predicted_actions": actions,  # Return GT during training
        }

    def _inference_forward(
        self,
        features: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference forward pass with DDPM sampling."""
        B = features.shape[0]
        device = features.device

        num_steps = num_inference_steps or self.num_diffusion_steps

        # Start from pure noise
        x = torch.randn(B, self.action_dim * self.chunk_size, device=device)

        # Reverse diffusion process
        for t in reversed(range(num_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self._predict_noise(x, t_batch, features)

            # DDPM update step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            ) + torch.sqrt(beta) * noise

        # Reshape to action chunks
        predicted_actions = x.view(B, self.chunk_size, self.action_dim)

        return {"predicted_actions": predicted_actions}

    def _predict_noise(
        self,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise given noisy actions, timestep, and conditioning."""
        # Get embeddings
        time_emb = self.time_emb(t.float())
        cond_emb = self.cond_proj(features)

        # Concatenate inputs
        x = torch.cat([noisy_actions, time_emb, cond_emb], dim=-1)

        # Predict noise
        return self.noise_pred(x)

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        num_inference_steps: int = 20,
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """
        Sample actions using DDPM or DDIM sampling.

        Args:
            features: (batch, input_dim) conditioning features
            num_inference_steps: Number of denoising steps (can be less than training)
            use_ddim: Whether to use DDIM (faster) or DDPM sampling

        Returns:
            actions: (batch, chunk_size, action_dim)
        """
        self.eval()
        if use_ddim and num_inference_steps < self.num_diffusion_steps:
            return self._ddim_sample(features, num_inference_steps)
        outputs = self._inference_forward(features, num_inference_steps)
        return outputs["predicted_actions"]

    def _ddim_sample(
        self,
        features: torch.Tensor,
        num_inference_steps: int = 20,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling for faster inference.

        Args:
            features: (batch, input_dim) conditioning features
            num_inference_steps: Number of denoising steps
            eta: DDIM stochasticity parameter (0 = deterministic)

        Returns:
            actions: (batch, chunk_size, action_dim)
        """
        B = features.shape[0]
        device = features.device

        # Create evenly spaced timesteps
        step_ratio = self.num_diffusion_steps // num_inference_steps
        timesteps = torch.arange(0, self.num_diffusion_steps, step_ratio, device=device)
        timesteps = torch.flip(timesteps, [0])

        # Start from pure noise
        x = torch.randn(B, self.action_dim * self.chunk_size, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self._predict_noise(x, t_batch, features)

            # Get alphas
            alpha_cumprod_t = self.alphas_cumprod[t.long()]

            if i + 1 < len(timesteps):
                alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i + 1].long()]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * pred_noise

            if i + 1 < len(timesteps):
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                x = pred_x0

        # Reshape to action chunks
        return x.view(B, self.chunk_size, self.action_dim)


if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Action Head Test")
    print("=" * 60)

    batch_size = 4
    input_dim = 1536
    action_dim = 7
    chunk_size = 10

    head = DiffusionActionHead(
        input_dim=input_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        num_diffusion_steps=100,
    )

    features = torch.randn(batch_size, input_dim)
    actions_gt = torch.randn(batch_size, chunk_size, action_dim)

    # Training mode
    print("\nTraining mode:")
    head.train()
    outputs = head(features, actions_gt)
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Inference mode
    print("\nInference mode (20 steps):")
    head.eval()
    with torch.no_grad():
        outputs = head(features)
    print(f"  Output shape: {outputs['predicted_actions'].shape}")

    # Fast sampling
    print("\nFast sampling (10 steps):")
    sampled = head.sample(features, num_inference_steps=10)
    print(f"  Output shape: {sampled.shape}")

    print("\n" + "=" * 60)
    print("Diffusion action head test passed!")
    print("=" * 60)
