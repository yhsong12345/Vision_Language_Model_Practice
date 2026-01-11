"""
Transformer Action Head for VLA Models

Autoregressive action prediction using transformer decoder.
Supports:
- Action chunking with temporal modeling
- Autoregressive generation
- Attention-based action prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class TransformerActionHead(nn.Module):
    """
    Transformer-based Action Prediction Head.

    Uses a transformer decoder to autoregressively predict action sequences.
    Suitable for:
    - Long action sequences
    - Temporal dependencies between actions
    - Attention-based reasoning over action history

    Args:
        input_dim: Conditioning feature dimension
        action_dim: Dimension of action space
        hidden_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        chunk_size: Maximum action sequence length
        dropout: Dropout probability
        use_causal_mask: Whether to use causal attention
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        chunk_size: int = 10,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.use_causal_mask = use_causal_mask

        # Condition projection
        self.cond_proj = nn.Linear(input_dim, hidden_dim)

        # Action embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            self._create_positional_encoding(chunk_size + 1, hidden_dim)
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int,
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _create_causal_mask(
        self,
        size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

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
        """Training forward pass with teacher forcing."""
        B = features.shape[0]
        device = features.device
        T = actions.shape[1]

        # Project conditioning
        cond = self.cond_proj(features).unsqueeze(1)  # (B, 1, hidden_dim)

        # Embed actions and prepend start token
        action_emb = self.action_embed(actions)  # (B, T, hidden_dim)
        start = self.start_token.expand(B, -1, -1)
        decoder_input = torch.cat([start, action_emb[:, :-1]], dim=1)  # (B, T, hidden_dim)

        # Add positional encoding
        decoder_input = decoder_input + self.pos_encoding[:, :T, :]

        # Create causal mask
        if self.use_causal_mask:
            causal_mask = self._create_causal_mask(T, device)
        else:
            causal_mask = None

        # Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=cond,
            tgt_mask=causal_mask,
        )

        # Project to action space
        predicted_actions = self.output_proj(decoder_output)

        # Compute loss
        loss = F.mse_loss(predicted_actions, actions)

        return {
            "predicted_actions": predicted_actions,
            "loss": loss,
        }

    def _inference_forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive inference."""
        B = features.shape[0]
        device = features.device

        # Project conditioning
        cond = self.cond_proj(features).unsqueeze(1)

        # Start with start token
        generated = [self.start_token.expand(B, -1, -1)]
        actions = []

        for t in range(self.chunk_size):
            # Current sequence
            decoder_input = torch.cat(generated, dim=1)
            T = decoder_input.shape[1]

            # Add positional encoding
            decoder_input = decoder_input + self.pos_encoding[:, :T, :]

            # Create causal mask
            if self.use_causal_mask:
                causal_mask = self._create_causal_mask(T, device)
            else:
                causal_mask = None

            # Decode
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=cond,
                tgt_mask=causal_mask,
            )

            # Get last output and project to action
            action = self.output_proj(decoder_output[:, -1:, :])
            actions.append(action)

            # Embed action for next step
            action_emb = self.action_embed(action)
            generated.append(action_emb)

        # Concatenate all actions
        predicted_actions = torch.cat(actions, dim=1)

        return {"predicted_actions": predicted_actions}

    @torch.no_grad()
    def generate(
        self,
        features: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate action sequence.

        Args:
            features: (batch, input_dim) conditioning features
            num_steps: Number of actions to generate (default: chunk_size)

        Returns:
            actions: (batch, num_steps, action_dim)
        """
        self.eval()

        if num_steps is not None:
            original_chunk_size = self.chunk_size
            self.chunk_size = num_steps

        outputs = self._inference_forward(features)

        if num_steps is not None:
            self.chunk_size = original_chunk_size

        return outputs["predicted_actions"]


class GPTActionHead(nn.Module):
    """
    GPT-style Action Head.

    Uses a decoder-only transformer for action prediction.
    More similar to language model architecture.

    Args:
        input_dim: Conditioning feature dimension
        action_dim: Dimension of action space
        hidden_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        chunk_size: Maximum action sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        chunk_size: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size

        # Input projections
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Positional embedding (learned)
        self.pos_embed = nn.Embedding(chunk_size + 1, hidden_dim)  # +1 for condition

        # Type embedding (condition vs action)
        self.type_embed = nn.Embedding(2, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B = features.shape[0]
        device = features.device

        # Condition token
        cond = self.cond_proj(features).unsqueeze(1)  # (B, 1, hidden_dim)
        cond = cond + self.type_embed(torch.zeros(1, dtype=torch.long, device=device))
        cond = cond + self.pos_embed(torch.zeros(1, dtype=torch.long, device=device))

        if self.training and actions is not None:
            T = actions.shape[1]

            # Action tokens
            action_emb = self.action_proj(actions)  # (B, T, hidden_dim)

            # Add type and position embeddings
            positions = torch.arange(1, T + 1, device=device)
            action_emb = action_emb + self.type_embed(torch.ones(1, dtype=torch.long, device=device))
            action_emb = action_emb + self.pos_embed(positions)

            # Concatenate [condition, shifted_actions]
            # For training, we predict action[t] from action[0:t-1]
            x = torch.cat([cond, action_emb[:, :-1]], dim=1)

            # Causal mask
            seq_len = x.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            # Forward through blocks
            for block in self.blocks:
                x = block(x, src_mask=mask.float().masked_fill(mask, float('-inf')))

            x = self.norm(x)
            predicted = self.output_proj(x[:, 1:])  # Skip condition token output

            loss = F.mse_loss(predicted, actions[:, :predicted.shape[1]])

            return {
                "predicted_actions": predicted,
                "loss": loss,
            }
        else:
            # Autoregressive generation
            x = cond
            actions_list = []

            for t in range(self.chunk_size):
                seq_len = x.shape[1]
                mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

                h = x
                for block in self.blocks:
                    h = block(h, src_mask=mask.float().masked_fill(mask, float('-inf')))

                h = self.norm(h)
                action = self.output_proj(h[:, -1:])
                actions_list.append(action)

                # Prepare next input
                action_emb = self.action_proj(action)
                action_emb = action_emb + self.type_embed(torch.ones(1, dtype=torch.long, device=device))
                action_emb = action_emb + self.pos_embed(torch.tensor([t + 1], device=device))
                x = torch.cat([x, action_emb], dim=1)

            predicted_actions = torch.cat(actions_list, dim=1)
            return {"predicted_actions": predicted_actions}


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Action Head Test")
    print("=" * 60)

    batch_size = 4
    input_dim = 1536
    action_dim = 7
    chunk_size = 10

    # Test Transformer Action Head
    print("\nTransformer Action Head:")
    head = TransformerActionHead(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_layers=4,
        chunk_size=chunk_size,
    )

    features = torch.randn(batch_size, input_dim)
    actions_gt = torch.randn(batch_size, chunk_size, action_dim)

    # Training
    head.train()
    outputs = head(features, actions_gt)
    print(f"  Training - Output: {outputs['predicted_actions'].shape}, Loss: {outputs['loss'].item():.4f}")

    # Inference
    head.eval()
    with torch.no_grad():
        outputs = head(features)
    print(f"  Inference - Output: {outputs['predicted_actions'].shape}")

    # Test GPT Action Head
    print("\nGPT Action Head:")
    gpt_head = GPTActionHead(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_layers=4,
        chunk_size=chunk_size,
    )

    gpt_head.train()
    outputs = gpt_head(features, actions_gt)
    print(f"  Training - Output: {outputs['predicted_actions'].shape}, Loss: {outputs['loss'].item():.4f}")

    gpt_head.eval()
    with torch.no_grad():
        outputs = gpt_head(features)
    print(f"  Inference - Output: {outputs['predicted_actions'].shape}")

    print("\n" + "=" * 60)
    print("All transformer action head tests passed!")
    print("=" * 60)
