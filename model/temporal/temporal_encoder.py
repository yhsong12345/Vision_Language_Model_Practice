"""
Temporal Encoder Module

Encodes sequences of observations over time for temporal reasoning.
Supports multiple architectures:
- Transformer-based temporal encoding
- LSTM-based temporal encoding
- Causal attention for autoregressive modeling
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from model.utils.layers import PositionalEncoding


@dataclass
class TemporalEncoderConfig:
    """Configuration for temporal encoder."""
    input_dim: int = 768
    hidden_dim: int = 512
    output_dim: int = 768
    num_layers: int = 4
    num_heads: int = 8
    max_seq_len: int = 64
    dropout: float = 0.1
    use_causal_mask: bool = True
    encoder_type: str = "transformer"  # "transformer" or "lstm"
    use_positional_encoding: bool = True
    use_learned_pos_embed: bool = True


class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal encoder.

    Uses self-attention to capture temporal dependencies across observations.
    Supports causal masking for autoregressive generation.
    """

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        if config.use_positional_encoding:
            if config.use_learned_pos_embed:
                self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)
            else:
                self.pos_encoding = PositionalEncoding(
                    config.hidden_dim, config.max_seq_len, config.dropout
                )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)

        if self.config.use_positional_encoding:
            if self.config.use_learned_pos_embed:
                positions = torch.arange(seq_len, device=x.device)
                x = x + self.pos_embed(positions)
            else:
                x = self.pos_encoding(x)

        causal_mask = None
        if self.config.use_causal_mask:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        x = self.output_proj(x)
        return self.layer_norm(x)

    def get_last_hidden_state(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.forward(x, attention_mask)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(output.size(0), device=output.device)
            return output[batch_indices, seq_lengths]
        return output[:, -1]


class TemporalLSTM(nn.Module):
    """LSTM-based temporal encoder for long sequences."""

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.input_proj(x)
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.output_proj(x)
        return self.layer_norm(x), hidden_state

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
        c_0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
        return (h_0, c_0)


class TemporalEncoder(nn.Module):
    """Unified temporal encoder interface wrapping Transformer or LSTM."""

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config

        if config.encoder_type == "transformer":
            self.encoder = TemporalTransformer(config)
        elif config.encoder_type == "lstm":
            self.encoder = TemporalLSTM(config)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")

        self.aggregation = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.config.encoder_type == "transformer":
            hidden_states = self.encoder(observations, attention_mask)
            last_hidden_state = self.encoder.get_last_hidden_state(observations, attention_mask)
        else:
            hidden_states, _ = self.encoder(observations)
            last_hidden_state = hidden_states[:, -1]

        aggregated = self.aggregation(last_hidden_state)

        return {
            "hidden_states": hidden_states if return_sequence else aggregated,
            "last_hidden_state": last_hidden_state,
            "aggregated": aggregated,
        }

    @classmethod
    def from_pretrained(cls, path: str) -> "TemporalEncoder":
        checkpoint = torch.load(path, map_location="cpu")
        config = TemporalEncoderConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def save_pretrained(self, path: str):
        torch.save({"config": vars(self.config), "state_dict": self.state_dict()}, path)


if __name__ == "__main__":
    config = TemporalEncoderConfig(input_dim=768, hidden_dim=512, output_dim=768, num_layers=4, max_seq_len=64)

    transformer_encoder = TemporalTransformer(config)
    x = torch.randn(4, 32, 768)
    mask = torch.ones(4, 32)
    print(f"Transformer output: {transformer_encoder(x, mask).shape}")

    config.encoder_type = "lstm"
    lstm_encoder = TemporalLSTM(config)
    output, _ = lstm_encoder(x)
    print(f"LSTM output: {output.shape}")

    config.encoder_type = "transformer"
    encoder = TemporalEncoder(config)
    outputs = encoder(x, mask)
    print(f"Unified encoder: hidden_states={outputs['hidden_states'].shape}, aggregated={outputs['aggregated'].shape}")
