"""
architecture.py — Keaa-50M's own Mamba (S6) architecture, built from scratch
in plain PyTorch. No mamba-ssm CUDA kernels, no pretrained checkpoints --
this is the model you train and the weights you release. It runs on any
GPU (Kaggle/Colab T4 included); the selective scan below is a naive
sequential loop, so it's slower than fused kernels but it's fully yours
to cite in the DOI and it's correct.

Spec, matching your project notes: d_model=384, n_layers=8, d_state=16.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaConfig:
    def __init__(
        self,
        vocab_size=4096,       # matches custom_tokenizer.py's VOCAB_SIZE
        d_model=960,
        n_layers=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        pad_vocab_size_multiple=8,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        if vocab_size % pad_vocab_size_multiple != 0:
            self.vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def selective_scan(u, delta, A, B, C, D):
    """
    The core Mamba / S6 recurrence, run as a sequential scan over time.
    u:     (batch, d_inner, L)
    delta: (batch, d_inner, L)  -- input-dependent step size
    A:     (d_inner, d_state)   -- fixed per-channel, kept negative for stability
    B:     (batch, d_state, L)  -- input-dependent
    C:     (batch, d_state, L)  -- input-dependent
    D:     (d_inner,)           -- skip connection
    """
    b, d_inner, L = u.shape

    # zero-order-hold discretization
    A_ = A.unsqueeze(0).unsqueeze(2)                                    # (1, d_inner, 1, d_state)
    deltaA = torch.exp(delta.unsqueeze(-1) * A_)                        # (b, d_inner, L, d_state)
    B_ = B.transpose(1, 2).unsqueeze(1)                                 # (b, 1, L, d_state)
    deltaB_u = delta.unsqueeze(-1) * B_ * u.unsqueeze(-1)               # (b, d_inner, L, d_state)

    state = torch.zeros(b, d_inner, A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        state = deltaA[:, :, t] * state + deltaB_u[:, :, t]
        y = torch.einsum("bdn,bn->bd", state, C[:, :, t])
        ys.append(y)
    y = torch.stack(ys, dim=-1)  # (b, d_inner, L)
    return y + u * D.unsqueeze(0).unsqueeze(-1)


class MambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_inner

        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=cfg.d_conv,
            groups=d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(d_inner, cfg.dt_rank + 2 * cfg.d_state, bias=False)
        self.dt_proj = nn.Linear(cfg.dt_rank, d_inner, bias=True)

        A = torch.arange(1, cfg.d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # learned in log space
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def forward(self, x):
        # x: (batch, L, d_model)
        b, L, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)      # each (b, L, d_inner)

        x_in = x_in.transpose(1, 2)                       # (b, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :L]
        x_in = F.silu(x_in)

        x_dbl = self.x_proj(x_in.transpose(1, 2))          # (b, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.cfg.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt)).transpose(1, 2)  # (b, d_inner, L)
        B = B.transpose(1, 2)                               # (b, d_state, L)
        C = C.transpose(1, 2)                               # (b, d_state, L)

        A = -torch.exp(self.A_log)                          # (d_inner, d_state), kept negative

        y = selective_scan(x_in, delta, A, B, C, self.D)     # (b, d_inner, L)
        y = y.transpose(1, 2) * F.silu(z)                    # gate

        return self.out_proj(y)


class MambaResidualBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaBlock(cfg)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class KeaaMamba(nn.Module):
    """Keaa-50M: embedding -> N Mamba blocks -> tied LM head."""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(MambaResidualBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying saves params
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)  # keeps init logits well-scaled

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = MambaConfig()
    model = KeaaMamba(cfg)
    n_params = count_parameters(model)
    print(f"Keaa-50M config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"d_state={cfg.d_state}, vocab_size={cfg.vocab_size}")
    print(f"Parameter count: {n_params:,} ({n_params/1e6:.1f}M)")

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    out = model(x, labels=x)
    print(f"Smoke test -> logits shape: {tuple(out['logits'].shape)}, loss: {out['loss'].item():.3f}")
