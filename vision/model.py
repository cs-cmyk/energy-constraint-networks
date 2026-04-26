"""
Constraint Network: SSM backbone + 2-head causal/bidirectional attention
Energy-based model for detecting structural consistency in sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleSSMBlock(nn.Module):
    """
    SSM block using 1D convolution + gating as a CPU-friendly approximation.
    Captures sequential dependencies without the slow autograd-through-loop.
    On GPU, swap this for a proper selective scan (Mamba).
    """

    def __init__(self, d_model, d_state=64, dropout=0.1, kernel_size=8):
        super().__init__()
        self.d_model = d_model

        # Depthwise causal convolution (captures local sequential structure)
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=d_model
        )

        # Gated linear unit for selective information flow
        self.gate_proj = nn.Linear(d_model, d_model * 2)

        # Learned exponential decay (simulates SSM state dynamics)
        self.decay = nn.Parameter(torch.randn(d_model) * 0.01)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        residual = x
        x = self.norm(x)
        batch, seq_len, d = x.shape

        # Causal convolution (local sequential patterns)
        h = x.transpose(1, 2)  # (B, d, L)
        h = self.conv(h)[:, :, :seq_len]  # trim to causal
        h = h.transpose(1, 2)  # (B, L, d)

        # Apply learned decay weighting (approximates SSM state evolution)
        decay_weights = torch.sigmoid(self.decay)  # (d,)
        # Cumulative decay effect along sequence
        positions = torch.arange(seq_len, device=x.device).float().unsqueeze(1)  # (L, 1)
        decay_mask = decay_weights.unsqueeze(0).pow(positions / seq_len)  # (L, d)
        h = h * decay_mask.unsqueeze(0)

        # Gated activation
        gate_out = self.gate_proj(h)
        h, gate = gate_out.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)

        h = self.out_proj(h)
        h = self.dropout(h)
        return h + residual


class TwoHeadAttention(nn.Module):
    """
    2-head attention block:
    - Head 1: causal masked (forward consistency)
    - Head 2: bidirectional (mutual compatibility)

    Single projection per head, no separate Q/K/V.
    h = Wx, attention = softmax(h @ h^T / sqrt(d_head)) @ h
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.d_head = d_model // 2
        self.d_model = d_model

        # One projection per head
        self.W1 = nn.Linear(d_model, self.d_head, bias=False)  # causal head
        self.W2 = nn.Linear(d_model, self.d_head, bias=False)  # bidirectional head

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        # Head 1: causal
        h1 = self.W1(x)  # (B, L, d_head)
        attn1 = torch.bmm(h1, h1.transpose(1, 2)) / self.scale
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        attn1 = attn1.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.dropout(attn1)
        out1 = torch.bmm(attn1, h1)  # (B, L, d_head)

        # Head 2: bidirectional
        h2 = self.W2(x)  # (B, L, d_head)
        attn2 = torch.bmm(h2, h2.transpose(1, 2)) / self.scale
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout(attn2)
        out2 = torch.bmm(attn2, h2)  # (B, L, d_head)

        # Concatenate and project
        out = torch.cat([out1, out2], dim=-1)  # (B, L, d_model)
        out = self.W_o(out)
        out = self.dropout(out)

        return out + residual


class ConstraintNetwork(nn.Module):
    """
    Full constraint network:
    6 SSM blocks + 2 interleaved attention blocks
    Outputs per-position energy + aggregated scalar energy.
    """

    def __init__(self, d_model=256, d_state=64, vocab_size=None, max_seq_len=512, dropout=0.1, alpha=0.3):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha

        # Input embedding (if working from tokens directly)
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None

        self.input_proj = nn.Linear(d_model, d_model)

        # SSM blocks + interleaved attention
        self.blocks = nn.ModuleList()
        for i in range(6):
            self.blocks.append(SimpleSSMBlock(d_model, d_state, dropout))
            if i in [1, 3]:
                self.blocks.append(TwoHeadAttention(d_model, dropout))

        # Energy head
        self.energy_norm = nn.LayerNorm(d_model)
        self.energy_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, return_per_position=False):
        """
        x: (batch, seq_len, d_model) or (batch, seq_len) if using embedding
        Returns: scalar energy per batch element
        """
        if self.embedding is not None and x.dtype == torch.long:
            x = self.embedding(x)

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # Per-position energy
        h = self.energy_norm(x)
        per_pos_energy = self.energy_mlp(h).squeeze(-1)  # (B, L)

        # Aggregate: mean + alpha * max
        mean_energy = per_pos_energy.mean(dim=1)
        max_energy = per_pos_energy.max(dim=1).values
        energy = mean_energy + self.alpha * max_energy

        if return_per_position:
            return energy, per_pos_energy
        return energy


class ConstraintLoss(nn.Module):
    """
    Contrastive energy loss:
    L = E(x_pos)^2 + max(0, margin - E(x_neg))^2
    """

    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin

    def forward(self, energy_pos, energy_neg):
        loss_pos = energy_pos.pow(2).mean()
        loss_neg = F.relu(self.margin - energy_neg).pow(2).mean()
        return loss_pos + loss_neg


if __name__ == "__main__":
    # Quick shape test
    model = ConstraintNetwork(d_model=128, d_state=32, vocab_size=1000)
    x = torch.randint(0, 1000, (4, 64))
    energy, per_pos = model(x, return_per_position=True)
    print(f"Input: {x.shape}")
    print(f"Scalar energy: {energy.shape} -> {energy}")
    print(f"Per-position energy: {per_pos.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
