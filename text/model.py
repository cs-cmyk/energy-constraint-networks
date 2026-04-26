"""
Full constraint network with proper selective scan SSM.
Use this on GPU for best results. Falls back to conv-based SSM on CPU.

Architecture:
  6 SSM blocks + 2 interleaved 2-head attention blocks
  2-head design: causal head + bidirectional head
  Single projection per head (no Q/K/V, no KV cache)
  Per-position + aggregated energy output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print("Using Mamba selective scan (GPU-optimized)")
except ImportError:
    HAS_MAMBA = False
    print("Mamba not available, using conv-based SSM fallback")


class ConvSSMBlock(nn.Module):
    """CPU-friendly SSM approximation using causal convolution + gating."""

    def __init__(self, d_model, d_state=64, dropout=0.1, kernel_size=8):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=kernel_size - 1, groups=d_model)
        self.gate_proj = nn.Linear(d_model, d_model * 2)
        self.decay = nn.Parameter(torch.randn(d_model) * 0.01)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, L, d = x.shape
        h = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        decay_weights = torch.sigmoid(self.decay)
        positions = torch.arange(L, device=x.device).float().unsqueeze(1)
        h = h * decay_weights.unsqueeze(0).pow(positions / L).unsqueeze(0)
        gate_out = self.gate_proj(h)
        h, gate = gate_out.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)
        return self.dropout(self.out_proj(h)) + residual


class MambaSSMBlock(nn.Module):
    """GPU-optimized SSM using Mamba selective scan."""

    def __init__(self, d_model, d_state=64, dropout=0.1, **kwargs):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        return self.dropout(self.mamba(x)) + residual


# Select SSM implementation
SSMBlock = MambaSSMBlock if HAS_MAMBA else ConvSSMBlock


class TwoHeadAttention(nn.Module):
    """
    2-head attention: causal + bidirectional.
    Single projection per head. No Q/K/V. No caching needed.
    
    Head 1 (causal): captures "is what follows consistent with what came before"
    Head 2 (bidirectional): captures "are any two positions mutually incompatible"
    
    Asymmetry emerges from:
    1. Different masks (causal vs full)
    2. Different learned projections (W1 vs W2)
    3. Output projection learns to combine directional + global signals
    
    Total params: 2 * d² (vs 4 * d² for standard Q/K/V attention)
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.d_head = d_model // 2
        self.W1 = nn.Linear(d_model, self.d_head, bias=False)
        self.W2 = nn.Linear(d_model, self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        # Head 1: causal
        h1 = self.W1(x)
        a1 = torch.bmm(h1, h1.transpose(1, 2)) / self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        a1 = a1.masked_fill(mask.unsqueeze(0), float("-inf"))
        out1 = torch.bmm(self.dropout(F.softmax(a1, dim=-1)), h1)

        # Head 2: bidirectional
        h2 = self.W2(x)
        a2 = torch.bmm(h2, h2.transpose(1, 2)) / self.scale
        out2 = torch.bmm(self.dropout(F.softmax(a2, dim=-1)), h2)

        out = self.W_o(torch.cat([out1, out2], dim=-1))
        return self.dropout(out) + residual


class ConstraintNetwork(nn.Module):
    """
    Energy-based constraint network.
    
    Architecture:
        Embedding -> [SSM, SSM, Attention, SSM, SSM, Attention, SSM, SSM] -> Energy head
    
    Output:
        Scalar energy E(x) = mean(e_i) + alpha * max(e_i)
        Low energy = structurally coherent
        High energy = structural violations detected
    
    Key properties:
        - No KV cache needed (evaluator, not generator)
        - SSM provides O(L) sequential processing
        - Sparse attention provides long-range violation detection
        - Per-position energy localizes violations
        - Stateless forward pass (clean for training inner loop)
    """

    def __init__(self, d_model=128, d_state=64, vocab_size=None,
                 max_seq_len=512, dropout=0.1, alpha=0.3):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None

        self.input_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList()
        for i in range(6):
            self.blocks.append(SSMBlock(d_model, d_state, dropout))
            if i in [1, 3]:
                self.blocks.append(TwoHeadAttention(d_model, dropout))

        self.energy_norm = nn.LayerNorm(d_model)
        self.energy_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, return_per_position=False):
        if self.embedding is not None and x.dtype == torch.long:
            x = self.embedding(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        per_pos = self.energy_mlp(self.energy_norm(x)).squeeze(-1)
        energy = per_pos.mean(dim=1) + self.alpha * per_pos.max(dim=1).values

        return (energy, per_pos) if return_per_position else energy


class ConstraintLoss(nn.Module):
    """
    Contrastive energy loss with squared terms.
    L = E(x_pos)² + max(0, margin - E(x_neg))²
    
    Squared terms force:
    - Coherent sequences toward zero energy (not just "low")
    - Corrupted sequences toward specific margin (smooth landscape)
    """

    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin

    def forward(self, energy_pos, energy_neg):
        return energy_pos.pow(2).mean() + F.relu(self.margin - energy_neg).pow(2).mean()
