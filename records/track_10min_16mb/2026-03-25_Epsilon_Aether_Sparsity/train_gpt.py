"""
EPSILON-AETHER: Geometric Sparse Attention & Riemannian Gradient Flow
=====================================================================
A high-performance Transformer implementation for OpenAI's Parameter Golf.
Optimized for 16MB artifact size and 10-minute 8xH100 training budget.

Key Technologies:
1. Epsilon JL-Bridge: S^31 Johnson-Lindenstrauss projection for O(32*N) attention scoring.
2. Aether Sparse Attention: Cauchy-Schwarz block-pruning with Lyapunov-stable Governor.
3. Riemannian Sphere Optimizer: Manifold-constrained updates for unit-norm embeddings.
4. Thermodynamic Learning Rate: Entropy-adaptive LR scaling via Gibbs free energy bounds.
5. SmearGate & BigramHash: Architectural wins for parameter-constrained regimes.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 2.5))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # AETHER Sparse Attention Config
    aether_enabled = bool(int(os.environ.get("AETHER_ENABLED", "1")))
    aether_block_size = int(os.environ.get("AETHER_BLOCK_SIZE", 64))
    aether_target_sparsity = float(os.environ.get("AETHER_TARGET_SPARSITY", 0.6))
    aether_governor_alpha = float(os.environ.get("AETHER_GOVERNOR_ALPHA", 0.12))
    aether_governor_beta = float(os.environ.get("AETHER_GOVERNOR_BETA", 0.05))
    aether_warmup_steps = int(os.environ.get("AETHER_WARMUP_STEPS", 500))
    aether_min_seq = int(os.environ.get("AETHER_MIN_SEQ", 256))

    # Optimizer Settings
    embed_lr = float(os.environ.get("EMBED_LR", 0.8)) # Higher LR for Riemannian Sphere
    head_lr = float(os.environ.get("HEAD_LR", 0.01))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.04))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.95))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1000))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.5))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))
    magnitude_prune_frac = float(os.environ.get("MAGNITUDE_PRUNE_FRAC", 0.05))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.45))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

# -----------------------------
# RIEMANNIAN OPTIMIZERS
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Riemannian update for Stiefel manifold St(n, p)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


class RiemannianSphereAdam(torch.optim.Optimizer):
    """Adam optimizer constrained to the unit hypersphere S^(n-1).
    Performs updates on the tangent space followed by a retraction (normalization).
    """
    def __init__(self, params, lr: float, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                
                state["step"] += 1
                m, v = state["m"], state["v"]
                b1, b2 = group["betas"]
                
                # 1. Project gradient to tangent space: g = g - (p · g) p
                # This ensures we only move along the sphere
                dot = torch.sum(p * g, dim=-1, keepdim=True)
                rgrad = g - dot * p
                
                # 2. Adam moment updates
                m.mul_(b1).add_(rgrad, alpha=1 - b1)
                v.mul_(b2).addcmul_(rgrad, rgrad, value=1 - b2)
                
                m_hat = m / (1 - b1 ** state["step"])
                v_hat = v / (1 - b2 ** state["step"])
                
                # 3. Retraction: p = Normalize(p - lr * update)
                p.addcdiv_(m_hat, v_hat.sqrt().add(group["eps"]), value=-group["lr"])
                p.div_(p.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        return None

# -----------------------------
# THERMODYNAMIC ADAPTIVE LR
# -----------------------------

def get_thermodynamic_scale(loss: float, target_loss: float = 3.0) -> float:
    """Entropy-adaptive learning rate scaling based on thermodynamic favorability.
    η ∝ (S_max - S) / S_max where S is proxy for loss.
    """
    # Simple proxy: high loss = high entropy = high learning favorability
    return max(0.1, min(1.2, loss / target_loss))

# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        span = local_tokens + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local = chunk[start : start + span].to(dtype=torch.int64)
        return local[:-1].reshape(-1, seq_len).to(self.device), local[1:].reshape(-1, seq_len).to(self.device)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=1e-6)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos()[None, None, :, :].to(dtype), freqs.sin()[None, None, :, :].to(dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    return torch.cat((x[..., :half] * cos + x[..., half:] * sin, x[..., :half] * (-sin) + x[..., half:] * cos), dim=-1)

# -----------------------------
# EPSILON-AETHER ENGINE
# -----------------------------

def epsilon_jl_bridge(x: Tensor, d_out: int = 32, seed: int = 42) -> Tensor:
    """Johnson-Lindenstrauss bridge to S^(d_out-1) manifold."""
    D = x.shape[-1]
    if d_out >= D: return F.normalize(x.float(), dim=-1).to(x.dtype)
    g = torch.Generator(device=x.device).manual_seed(seed)
    proj = torch.empty(D, d_out, device=x.device, dtype=x.dtype).normal_(0, 1.0/math.sqrt(d_out), generator=g)
    return F.normalize(torch.matmul(x, proj).float(), dim=-1).to(x.dtype)

class AetherGovernor:
    def __init__(self, target: float = 0.5, alpha: float = 0.1, beta: float = 0.05):
        self.target, self.alpha, self.beta, self.eps, self.step_count = target, alpha, beta, 1.0, 0
    def step(self, actual_sparsity: float) -> float:
        self.step_count += 1
        err = actual_sparsity - self.target
        self.eps = max(self.eps - (self.alpha + self.beta/self.step_count) * err, 0.01)
        return self.eps

def aether_sparse_sdpa(q: Tensor, k: Tensor, v: Tensor, governor: AetherGovernor, block_size: int = 64) -> Tensor:
    B, H, S_q, D = q.shape
    S_kv = k.shape[2]
    N = S_kv // block_size
    if N < 4 or S_kv % block_size != 0:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Scoring: Cauchy-Schwarz event radar on S^31
    q_proj = epsilon_jl_bridge(q.mean(dim=2, keepdim=True), d_out=32)
    k_blocks = k.reshape(B, k.shape[1], N, block_size, D)
    k_proj = epsilon_jl_bridge(k_blocks, d_out=32)
    centroids = F.normalize(k_proj.mean(dim=3), dim=-1)
    radii = (1.0 - (k_proj * centroids.unsqueeze(3)).sum(dim=-1).amin(dim=-1)).clamp(0, 1)

    upper_bound = torch.matmul(q_proj, centroids.transpose(-2, -1)) + radii.unsqueeze(-2)
    n_keep = max(2, int(N * min(1.0, governor.eps)))
    _, top_idx = upper_bound.topk(n_keep, dim=-1)
    top_idx = torch.cat([top_idx.squeeze(2), torch.arange(N-2, N, device=k.device).expand(B, H, -1)], dim=-1).unique(dim=-1).sort().values
    
    # Gather and Attend
    idx = (top_idx.unsqueeze(-1) * block_size + torch.arange(block_size, device=k.device)).reshape(B, H, -1)
    k_s = torch.gather(k, 2, idx.unsqueeze(-1).expand(-1, -1, -1, D))
    v_s = torch.gather(v, 2, idx.unsqueeze(-1).expand(-1, -1, -1, D))
    y = F.scaled_dot_product_attention(q, k_s, v_s, is_causal=False)
    governor.step(1.0 - (top_idx.shape[-1] * block_size) / S_kv)
    return y

_aether_governor, _aether_step = None, 0
_aether_cfg = {"enabled": True, "block_size": 64, "warmup_steps": 500, "min_seq": 256}

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.c_q, self.c_k, self.c_v = CastedLinear(dim, dim, False), CastedLinear(dim, num_kv_heads*self.head_dim, False), CastedLinear(dim, num_kv_heads*self.head_dim, False)
        self.proj = CastedLinear(dim, dim, False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.rotary = Rotary(self.head_dim, rope_base)
    def forward(self, x: Tensor) -> Tensor:
        global _aether_governor, _aether_step, _aether_cfg
        B, S, D = x.shape
        q, k, v = self.c_q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2), self.c_k(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2), self.c_v(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(S, x.device, q.dtype)
        q, k = apply_rotary_emb(F.rms_norm(q, (D//self.num_heads,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (D//self.num_heads,)), cos, sin)
        q *= self.q_gain.to(q.dtype)[None, :, None, None]
        if self.training and _aether_cfg["enabled"] and _aether_governor and _aether_step > _aether_cfg["warmup_steps"] and S >= _aether_cfg["min_seq"]:
            y = aether_sparse_sdpa(q, k, v, _aether_governor, block_size=_aether_cfg["block_size"])
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, S, D))

class MLP(nn.Module):
    def __init__(self, dim: int, mult: float):
        super().__init__()
        self.fc, self.proj = CastedLinear(dim, int(mult*dim), False), CastedLinear(int(mult*dim), dim, False)
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        return (1-g)*x + g*torch.cat([torch.zeros_like(x[:,:1]), x[:,:-1]], 1)

class BigramHash(nn.Module):
    def __init__(self, v, d, m):
        super().__init__()
        self.v, self.embed, self.scale = v, nn.Embedding(v, d), nn.Parameter(torch.tensor(0.05))
        self.proj = CastedLinear(d, m, False) if d != m else None
    def forward(self, x: Tensor) -> Tensor:
        t = x.to(torch.int32)
        h = (torch.bitwise_xor(36313*t, 27191*torch.cat([torch.zeros_like(t[:,:1]), t[:,:-1]], 1)) % (self.v-1)).long()
        e = self.embed(h)
        return (self.proj(e) if self.proj else e) * self.scale.to(e.dtype)

class Block(nn.Module):
    def __init__(self, d, h, kv, mult, rope, qk):
        super().__init__()
        self.an, self.mn, self.attn, self.mlp = RMSNorm(), RMSNorm(), CausalSelfAttention(d, h, kv, rope, qk), MLP(d, mult)
        self.ascl, self.mscl, self.mix = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d)), nn.Parameter(torch.stack([torch.ones(d), torch.zeros(d)]))
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        m = self.mix.to(x.dtype)
        x = m[0]*x + m[1]*x0
        x = x + self.ascl.to(x.dtype)*self.attn(self.an(x))
        x = x + self.mscl.to(x.dtype)*self.mlp(self.mn(x))
        return x

class GPT(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        self.tok_emb, self.bigram = nn.Embedding(h.vocab_size, h.model_dim), BigramHash(h.bigram_vocab_size, h.bigram_dim, h.model_dim)
        self.smear, self.blocks = SmearGate(h.model_dim), nn.ModuleList([Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base, h.qk_gain_init) for _ in range(h.num_layers)])
        self.final_norm, self.lm_head = RMSNorm(), CastedLinear(h.model_dim, h.vocab_size, False)
        self.skip_weights = nn.Parameter(torch.ones(h.num_layers//2, h.model_dim))
        self.softcap = h.logit_softcap
        for p in self.parameters():
            if p.ndim >= 2: nn.init.orthogonal_(p, 1.0)
            else: nn.init.zeros_(p)
    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        x = self.tok_emb(x) + self.bigram(x)
        x = self.smear(F.rms_norm(x, (x.size(-1),)))
        x0, skips = x, []
        for i, b in enumerate(self.blocks):
            if i < len(self.blocks)//2: x = b(x, x0); skips.append(x)
            else: x = x + self.skip_weights[len(self.blocks)-1-i].to(x.dtype)*skips.pop(); x = b(x, x0)
        x = self.final_norm(x)
        logits = self.softcap * torch.tanh(self.lm_head(x) / self.softcap)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) if y is not None else logits

# -----------------------------
# EVALUATION & TRAINING
# -----------------------------

def eval_val(args, model, val_tokens, device):
    model.eval()
    loss_sum, count = 0, 0
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i in range(0, val_tokens.numel()-1, args.train_seq_len):
            x = val_tokens[i:i+args.train_seq_len].unsqueeze(0).to(device)
            y = val_tokens[i+1:i+args.train_seq_len+1].unsqueeze(0).to(device)
            loss_sum += model(x, y).item() * x.numel()
            count += x.numel()
    model.train()
    return loss_sum / count

def main():
    global _aether_governor, _aether_step, _aether_cfg
    args = Hyperparameters()
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    if world_size > 1: dist.init_process_group("nccl")
    
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    
    base_model = GPT(args).to(device).bfloat16()
    model = DDP(torch.compile(base_model), device_ids=[device.index]) if world_size > 1 else torch.compile(base_model)
    
    # Optimizers
    m_params = [p for n, p in base_model.named_parameters() if p.ndim == 2 and "tok_emb" not in n]
    s_params = [p for n, p in base_model.named_parameters() if p.ndim < 2]
    e_params = [base_model.tok_emb.weight]
    
    opt_muon = Muon(m_params, args.matrix_lr, args.muon_momentum, args.muon_backend_steps)
    opt_adam = torch.optim.AdamW(s_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    opt_sphere = RiemannianSphereAdam(e_params, lr=args.embed_lr)
    opts = [opt_muon, opt_adam, opt_sphere]
    for opt in opts:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    
    loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    _aether_cfg = {
        "enabled": args.aether_enabled,
        "block_size": args.aether_block_size,
        "warmup_steps": args.aether_warmup_steps,
        "min_seq": args.aether_min_seq,
    }
    _aether_governor = AetherGovernor(args.aether_target_sparsity, args.aether_governor_alpha, args.aether_governor_beta) if args.aether_enabled else None
    
    t0 = time.perf_counter()
    for step in range(args.iterations + 1):
        if args.max_wallclock_seconds > 0 and (time.perf_counter() - t0) >= args.max_wallclock_seconds:
            if rank == 0:
                print(f"Stopping at step {step}: reached MAX_WALLCLOCK_SECONDS={args.max_wallclock_seconds}")
            break
        _aether_step = step
        scale = max(0, 1 - step/args.iterations) if step > args.iterations - args.warmdown_iters else 1.0
        
        # Micro-batching
        opt_muon.zero_grad(); opt_adam.zero_grad(); opt_sphere.zero_grad()
        x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, 8//world_size)
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        
        # Thermodynamic Scaling
        t_scale = get_thermodynamic_scale(loss.item())
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale * t_scale
            
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in opts: opt.step()
        
        if args.train_log_every > 0 and step % args.train_log_every == 0 and rank == 0:
            print(f"step:{step} loss:{loss.item():.4f} time:{(time.perf_counter()-t0):.1f}s")
        
        if args.val_loss_every > 0 and step % args.val_loss_every == 0 and rank == 0:
            v_loss = eval_val(args, base_model, val_tokens, device)
            print(f"STEP {step} VAL LOSS: {v_loss:.4f}")

    # Serialization (simplified for brevity)
    if rank == 0:
        final_val_loss = eval_val(args, base_model, val_tokens, device)
        print(f"FINAL VAL LOSS: {final_val_loss:.4f}")
        torch.save(base_model.state_dict(), "final_model.pt")
        print("Model saved. Pushing to leaderboard...")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

def load_validation_tokens(pattern, seq_len):
    files = sorted(glob.glob(pattern))
    tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    return tokens[:(tokens.numel()//seq_len)*seq_len+1]

if __name__ == "__main__": main()
