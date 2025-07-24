import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import warnings
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import datetime

warnings.filterwarnings('ignore')

def setup_results_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/muon_focused_ablation_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    (results_dir / "plots").mkdir(exist_ok=True)
    (results_dir / "data").mkdir(exist_ok=True)
    (results_dir / "reports").mkdir(exist_ok=True)
    (results_dir / "models").mkdir(exist_ok=True)
    
    return results_dir

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    batch_size: int = 32
    max_steps: int = 1500

    gradient_accumulation_steps: int = 2
    muon_lr: float = 0.01

    max_seq_len: int = 256
    num_documents: int = 1000
    max_tokens: int = 200000

    eval_every: int = 25
    eval_steps: int = 25

    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

# Newton-Schulz implementations
@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for i in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
        # Safety check for numerical stability
        if torch.isnan(X).any() or torch.isinf(X).any():
            X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

@torch.compile
def zeropower_conservative(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.0, -4.0, 1.5)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-6)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

@torch.compile
def zeropower_mild_aggressive(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Less aggressive than before to avoid NaN"""
    assert G.ndim >= 2
    a, b, c = (3.2, -4.2, 1.8)  # Further toned down for stability
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-6)

    for i in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
        # Check for NaN and clip if needed
        if torch.isnan(X).any():
            X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-6)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

@torch.compile
def zeropower_ultra_stable(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Ultra stable with strong regularization"""
    assert G.ndim >= 2
    a, b, c = (2.8, -3.5, 1.2)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-5)

    for i in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
        # Renormalize every step for ultra stability
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-5)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

# Focused Muon variants based on promising results
class MuonBase(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

# Newton-Schulz steps variations (focus on promising range)
class MuonSteps6(MuonBase):
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True):
        super().__init__(params, lr, momentum, nesterov, ns_steps=6)

class MuonSteps8(MuonBase):
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True):
        super().__init__(params, lr, momentum, nesterov, ns_steps=8)

class MuonSteps9(MuonBase):
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True):
        super().__init__(params, lr, momentum, nesterov, ns_steps=9)

class MuonSteps10(MuonBase):
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True):
        super().__init__(params, lr, momentum, nesterov, ns_steps=10)

class MuonSteps12(MuonBase):
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True):
        super().__init__(params, lr, momentum, nesterov, ns_steps=12)

# Momentum variations (focus on promising range)
class MuonMomentum88(MuonBase):
    def __init__(self, params, lr=0.01, nesterov=True, ns_steps=10):
        super().__init__(params, lr, momentum=0.88, nesterov=nesterov, ns_steps=ns_steps)

class MuonMomentum90(MuonBase):
    def __init__(self, params, lr=0.01, nesterov=True, ns_steps=10):
        super().__init__(params, lr, momentum=0.90, nesterov=nesterov, ns_steps=ns_steps)

class MuonMomentum92(MuonBase):
    def __init__(self, params, lr=0.01, nesterov=True, ns_steps=10):
        super().__init__(params, lr, momentum=0.92, nesterov=nesterov, ns_steps=ns_steps)

class MuonMomentum93(MuonBase):
    def __init__(self, params, lr=0.01, nesterov=True, ns_steps=10):
        super().__init__(params, lr, momentum=0.93, nesterov=nesterov, ns_steps=ns_steps)

class MuonMomentum96(MuonBase):
    def __init__(self, params, lr=0.01, nesterov=True, ns_steps=10):
        super().__init__(params, lr, momentum=0.96, nesterov=nesterov, ns_steps=ns_steps)

# Learning rate variations (fixed init)
class MuonLR005(MuonBase):
    def __init__(self, params, momentum=0.90, nesterov=True, ns_steps=10):
        super().__init__(params, lr=0.005, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

class MuonLR015(MuonBase):
    def __init__(self, params, momentum=0.90, nesterov=True, ns_steps=10):
        super().__init__(params, lr=0.015, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

class MuonLR02(MuonBase):
    def __init__(self, params, momentum=0.90, nesterov=True, ns_steps=10):
        super().__init__(params, lr=0.02, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

# Newton-Schulz variants
class MuonConservative10(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.90, nesterov=True, ns_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_conservative(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

class MuonMildAggressive10(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.90, nesterov=True, ns_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_mild_aggressive(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

class MuonUltraStable10(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.90, nesterov=True, ns_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_ultra_stable(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

# Hybrid approaches
class MuonBest10NoNesterov(MuonBase):
    def __init__(self, params, lr=0.01, momentum=0.90, ns_steps=10):
        super().__init__(params, lr, momentum, nesterov=False, ns_steps=ns_steps)

class MuonOptimal(torch.optim.Optimizer):
    """Best combination found so far"""
    def __init__(self, params, lr=0.01, momentum=0.90, nesterov=True, ns_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

# Model architecture and training code (unchanged but included for completeness)
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size
        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_optimizer(model: nn.Module, config: ModelConfig, optimizer_class):
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and
            'token_embedding' not in name and
            'norm' not in name and
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_optimizer = optimizer_class(muon_params)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model_variant(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader,
                       optimizer_class, variant_name: str, results_dir: Path):
    print(f"\n🚀 Training with {variant_name}")

    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizers = setup_optimizer(model, config, optimizer_class)

    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    model.train()
    step = 0
    start_time = time.time()
    
    # Fixed training history tracking
    training_history = {
        'steps': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_perplexity': [],
        'learning_rates': [],
        'grad_norms': []
    }

    pbar = tqdm(total=config.max_steps, desc=f"Training {variant_name}")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            current_loss = loss.item() * config.gradient_accumulation_steps
            
            # Check for NaN loss and abort if found
            if math.isnan(current_loss) or math.isinf(current_loss):
                print(f"❌ NaN/Inf loss detected at step {step}, aborting training")
                break

            if (step + 1) % config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            else:
                grad_norm = torch.tensor(0.0)  # Default value when no gradient step

            # FIXED: Evaluate more frequently and ensure data is saved
            if step % config.eval_every == 0:
                val_metrics = evaluate_model(model, val_loader, config)
                
                training_history['steps'].append(step)
                training_history['train_loss'].append(current_loss)
                training_history['val_loss'].append(val_metrics['val_loss'])
                training_history['val_accuracy'].append(val_metrics['val_accuracy'])
                training_history['val_perplexity'].append(val_metrics['val_perplexity'])
                training_history['learning_rates'].append(optimizers[0].param_groups[0]['lr'])
                training_history['grad_norms'].append(grad_norm.item())

            if step % 50 == 0:
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})

            step += 1
            if step % 50 == 0:
                pbar.update(50)

    pbar.close()
    
    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    
    # Add final point if not already added
    if not training_history['steps'] or training_history['steps'][-1] != step - 1:
        training_history['steps'].append(step - 1)
        training_history['train_loss'].append(current_loss)
        training_history['val_loss'].append(final_eval['val_loss'])
        training_history['val_accuracy'].append(final_eval['val_accuracy'])
        training_history['val_perplexity'].append(final_eval['val_perplexity'])
        training_history['learning_rates'].append(optimizers[0].param_groups[0]['lr'])
        training_history['grad_norms'].append(0.0)  # Placeholder for final step
    
    training_time = time.time() - start_time

    # Save training history
    history_file = results_dir / "data" / f"{variant_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)

    return {
        'variant': variant_name,
        'final_loss': training_history['train_loss'][-1] if training_history['train_loss'] else float('inf'),
        'val_loss': final_eval['val_loss'],
        'val_accuracy': final_eval['val_accuracy'],
        'val_perplexity': final_eval['val_perplexity'],
        'training_time': training_time,
        'training_history': training_history
    }

def create_plots(results: List[Dict], results_dir: Path):
    """Fixed plotting with better error handling"""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Filter out failed results
    valid_results = [r for r in results if not math.isnan(r.get('val_loss', float('nan')))]
    
    if not valid_results:
        print("❌ No valid results to plot")
        return
    
    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    variants = [r['variant'] for r in valid_results]
    val_losses = [r['val_loss'] for r in valid_results]
    val_perplexities = [r['val_perplexity'] for r in valid_results]
    val_accuracies = [r['val_accuracy'] for r in valid_results]
    training_times = [r['training_time'] for r in valid_results]
    
    sorted_indices = sorted(range(len(val_losses)), key=lambda i: val_losses[i])
    
    # Validation Loss
    axes[0, 0].bar(range(len(variants)), [val_losses[i] for i in sorted_indices], color='skyblue')
    axes[0, 0].set_title('Validation Loss Comparison')
    axes[0, 0].set_xlabel('Optimizer Variant')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_xticks(range(len(variants)))
    axes[0, 0].set_xticklabels([variants[i] for i in sorted_indices], rotation=45, ha='right')
    
    # Validation Perplexity
    axes[0, 1].bar(range(len(variants)), [val_perplexities[i] for i in sorted_indices], color='lightcoral')
    axes[0, 1].set_title('Validation Perplexity Comparison')
    axes[0, 1].set_xlabel('Optimizer Variant')
    axes[0, 1].set_ylabel('Validation Perplexity')
    axes[0, 1].set_xticks(range(len(variants)))
    axes[0, 1].set_xticklabels([variants[i] for i in sorted_indices], rotation=45, ha='right')
    
    # Validation Accuracy
    axes[1, 0].bar(range(len(variants)), [val_accuracies[i] for i in sorted_indices], color='lightgreen')
    axes[1, 0].set_title('Validation Accuracy Comparison')
    axes[1, 0].set_xlabel('Optimizer Variant')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].set_xticks(range(len(variants)))
    axes[1, 0].set_xticklabels([variants[i] for i in sorted_indices], rotation=45, ha='right')
    
    # Training Time
    axes[1, 1].bar(range(len(variants)), [training_times[i] for i in sorted_indices], color='gold')
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_xlabel('Optimizer Variant')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_xticks(range(len(variants)))
    axes[1, 1].set_xticklabels([variants[i] for i in sorted_indices], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. FIXED Training curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_results)))
    
    curves_plotted = 0
    for i, result in enumerate(valid_results):
        if 'training_history' in result and result['training_history']['steps']:
            history = result['training_history']
            
            # Ensure we have data
            if len(history['steps']) > 0 and len(history['train_loss']) > 0:
                try:
                    # Training Loss
                    axes[0, 0].plot(history['steps'], history['train_loss'], 
                                   label=result['variant'], color=colors[i], alpha=0.8, linewidth=2)
                    
                    # Validation Loss  
                    axes[0, 1].plot(history['steps'], history['val_loss'], 
                                   label=result['variant'], color=colors[i], alpha=0.8, linewidth=2)
                    
                    # Validation Accuracy
                    axes[1, 0].plot(history['steps'], history['val_accuracy'], 
                                   label=result['variant'], color=colors[i], alpha=0.8, linewidth=2)
                    
                    # Learning Rate
                    axes[1, 1].plot(history['steps'], history['learning_rates'], 
                                   label=result['variant'], color=colors[i], alpha=0.8, linewidth=2)
                    
                    curves_plotted += 1
                except Exception as e:
                    print(f"Warning: Could not plot curves for {result['variant']}: {e}")
    
    if curves_plotted > 0:
        axes[0, 0].set_title('Training Loss Curves')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss Curves')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Validation Accuracy Curves')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Validation Accuracy')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Learning Rate Schedules')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "plots" / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training curves plotted for {curves_plotted} variants")
    else:
        plt.close()
        print("❌ No training curves could be plotted")
    
    # 3. Heatmap
    if len(valid_results) > 1:
        df_data = []
        for result in valid_results:
            df_data.append({
                'Variant': result['variant'],
                'Val Loss': result['val_loss'],
                'Val Perplexity': result['val_perplexity'],
                'Val Accuracy': result['val_accuracy'],
                'Training Time': result['training_time']
            })
        
        df = pd.DataFrame(df_data)
        
        # Normalize metrics
        df_norm = df.copy()
        for col in ['Val Loss', 'Val Perplexity', 'Training Time']:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:
                df_norm[col] = 1 - (df[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = 0.5
        
        col_min, col_max = df['Val Accuracy'].min(), df['Val Accuracy'].max()
        if col_max > col_min:
            df_norm['Val Accuracy'] = (df['Val Accuracy'] - col_min) / (col_max - col_min)
        else:
            df_norm['Val Accuracy'] = 0.5
        
        plt.figure(figsize=(10, max(6, len(valid_results) * 0.5)))
        sns.heatmap(df_norm.set_index('Variant')[['Val Loss', 'Val Perplexity', 'Val Accuracy', 'Training Time']], 
                    annot=True, cmap='RdYlGn', center=0.5, 
                    fmt='.3f', cbar_kws={'label': 'Normalized Performance (Higher is Better)'})
        plt.title('Optimizer Performance Heatmap\n(Normalized Metrics)')
        plt.tight_layout()
        plt.savefig(results_dir / "plots" / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Plots saved to {results_dir / 'plots'}")

def generate_html_report(results: List[Dict], config: ModelConfig, results_dir: Path):
    """Generate comprehensive HTML report"""
    
    # Filter valid results
    valid_results = [r for r in results if not math.isnan(r.get('val_loss', float('nan')))]
    results_sorted = sorted(valid_results, key=lambda x: x['val_loss'])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Focused Muon Optimizer Ablation Study</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }}
            .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .best {{ background-color: #d5f4e6; padding: 15px; border-radius: 5px; border-left: 5px solid #27ae60; }}
            .failed {{ background-color: #ffeaa7; padding: 15px; border-radius: 5px; border-left: 5px solid #f39c12; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #2980b9; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .config {{ background-color: #fdf6e3; padding: 15px; border-radius: 5px; font-family: monospace; }}
            .insight {{ background-color: #e8f6ff; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Focused Muon Optimizer Ablation Study</h1>
            
            <div class="summary">
                <h2>📋 Experiment Configuration</h2>
                <div class="config">
                    Model Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H<br>
                    Training Steps: {config.max_steps}<br>
                    Batch Size: {config.batch_size}<br>
                    Sequence Length: {config.max_seq_len}<br>
                    Base Learning Rate: {config.muon_lr}<br>
                    Evaluation Frequency: Every {config.eval_every} steps<br>
                    Total Experiments: {len(results)} ({len(valid_results)} successful)
                </div>
            </div>
    """
    
    if results_sorted:
        html_content += f"""
            <div class="best">
                <h2>🏆 Best Performing Optimizer</h2>
                <strong>{results_sorted[0]['variant']}</strong><br>
                <span class="metric">Validation Loss:</span> {results_sorted[0]['val_loss']:.4f}<br>
                <span class="metric">Validation Perplexity:</span> {results_sorted[0]['val_perplexity']:.2f}<br>
                <span class="metric">Validation Accuracy:</span> {results_sorted[0]['val_accuracy']:.4f}<br>
                <span class="metric">Training Time:</span> {results_sorted[0]['training_time']:.1f}s
            </div>
        """
    
    # Show failed experiments
    failed_results = [r for r in results if math.isnan(r.get('val_loss', 0))]
    if failed_results:
        html_content += f"""
            <div class="failed">
                <h2>⚠️ Failed Experiments</h2>
                Failed variants: {', '.join([r['variant'] for r in failed_results])}<br>
                This indicates numerical instability or optimization issues.
            </div>
        """
    
    html_content += f"""
            <h2>📊 Detailed Results</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Optimizer Variant</th>
                    <th>Val Loss</th>
                    <th>Val Perplexity</th>
                    <th>Val Accuracy</th>
                    <th>Training Time (s)</th>
                    <th>Improvement vs Base</th>
                </tr>
    """
    
    # Find base Muon for comparison
    base_muon_loss = None
    for result in valid_results:
        if 'Base' in result['variant']:
            base_muon_loss = result['val_loss']
            break
    
    for i, result in enumerate(results_sorted):
        improvement = ""
        if base_muon_loss and result['val_loss'] != base_muon_loss:
            improvement_pct = ((base_muon_loss - result['val_loss']) / base_muon_loss) * 100
            improvement = f"{improvement_pct:+.1f}%"
        
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{result['variant']}</td>
                    <td>{result['val_loss']:.4f}</td>
                    <td>{result['val_perplexity']:.2f}</td>
                    <td>{result['val_accuracy']:.4f}</td>
                    <td>{result['training_time']:.1f}</td>
                    <td>{improvement}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>📈 Visualizations</h2>
            
            <div class="plot">
                <h3>Performance Comparison</h3>
                <img src="plots/performance_comparison.png" alt="Performance Comparison">
            </div>
            
            <div class="plot">
                <h3>Training Curves</h3>
                <img src="plots/training_curves.png" alt="Training Curves">
            </div>
            
            <div class="plot">
                <h3>Performance Heatmap</h3>
                <img src="plots/performance_heatmap.png" alt="Performance Heatmap">
            </div>
            
            <h2>🔍 Key Insights</h2>
    """
    
    # Generate insights
    if results_sorted:
        best_variant = results_sorted[0]['variant']
        fastest_variant = min(valid_results, key=lambda x: x['training_time'])['variant']
        
        # Analyze patterns
        step_variants = [r for r in valid_results if 'Steps' in r['variant']]
        momentum_variants = [r for r in valid_results if 'Momentum' in r['variant']]
        lr_variants = [r for r in valid_results if 'LR' in r['variant']]
        
        html_content += f"""
            <div class="insight">
                <strong>🥇 Overall Best:</strong> {best_variant} achieved the lowest validation loss of {results_sorted[0]['val_loss']:.4f}
            </div>
            
            <div class="insight">
                <strong>⚡ Speed Champion:</strong> {fastest_variant} completed training in {min(r['training_time'] for r in valid_results):.1f} seconds
            </div>
            
            <div class="insight">
                <strong>📈 Performance Range:</strong> Validation loss ranged from {min(r['val_loss'] for r in valid_results):.4f} to {max(r['val_loss'] for r in valid_results):.4f}
            </div>
        """
        
        if step_variants:
            best_steps = min(step_variants, key=lambda x: x['val_loss'])
            html_content += f"""
            <div class="insight">
                <strong>🔄 Newton-Schulz Steps:</strong> Best performing was {best_steps['variant']} with {best_steps['val_loss']:.4f} validation loss
            </div>
            """
        
        if momentum_variants:
            best_momentum = min(momentum_variants, key=lambda x: x['val_loss'])
            html_content += f"""
            <div class="insight">
                <strong>📊 Momentum Analysis:</strong> Best momentum variant was {best_momentum['variant']} with {best_momentum['val_loss']:.4f} validation loss
            </div>
            """
        
        if base_muon_loss:
            improvements = [(r['variant'], ((base_muon_loss - r['val_loss']) / base_muon_loss) * 100) 
                          for r in valid_results if r['val_loss'] < base_muon_loss]
            if improvements:
                best_improvement = max(improvements, key=lambda x: x[1])
                html_content += f"""
            <div class="insight">
                <strong>🚀 Best Improvement:</strong> {best_improvement[0]} improved upon base Muon by {best_improvement[1]:.1f}%
            </div>
                """
    
    html_content += f"""
            
            <div class="summary">
                <p><strong>Generated on:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Total Training Time:</strong> {sum(r['training_time'] for r in valid_results):.1f} seconds</p>
                <p><strong>Success Rate:</strong> {len(valid_results)}/{len(results)} experiments ({len(valid_results)/len(results)*100:.1f}%)</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(results_dir / "reports" / "focused_ablation_report.html", 'w') as f:
        f.write(html_content)
    
    print(f"📝 HTML report generated: {results_dir / 'reports' / 'focused_ablation_report.html'}")

if __name__ == "__main__":
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results_dir = setup_results_dir()
    print(f"📁 Results will be saved to: {results_dir}")

    set_seed(42)
    config = ModelConfig()

    print(f"\n📋 Focused Muon Ablation Study:")
    print(f"   🎯 Target: Beat validation loss ~5.52 (best Muon from previous run)")
    print(f"   📊 Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   ⏱️  Training: {config.max_steps} steps, eval every {config.eval_every} steps")

    # Load data (will use cache from previous run)
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Large-Scale Ablation Study
    experiments = []
    for lr in [1/32, 1/16, 1/8]:
        for momentum in [7/8, 15/16]:
            for ns_steps in [4, 8, 16]:
                variant_name = f"Muon (LR {lr:.4f}, Momentum {momentum:.4f}, {ns_steps} Steps)"
                experiments.append(
                    (lambda params, lr=lr, momentum=momentum, ns_steps=ns_steps: MuonBase(params, lr=lr, momentum=momentum, nesterov=True, ns_steps=ns_steps), variant_name)
                )

    results = []

    print(f"\n🧪 Running {len(experiments)} focused ablation experiments...")
    print(f"⏱️  Estimated total time: {len(experiments) * 1.5:.1f} minutes")

    for i, (optimizer_class, variant_name) in enumerate(experiments):
        try:
            print(f"\n[{i+1}/{len(experiments)}] Starting {variant_name}...")
            result = train_model_variant(config, train_loader, val_loader, optimizer_class, variant_name, results_dir)
            results.append(result)
            
            if not math.isnan(result['val_loss']):
                print(f"✅ {variant_name}: Val Loss: {result['val_loss']:.4f}, "
                      f"Val PPL: {result['val_perplexity']:.2f}, Time: {result['training_time']:.1f}s")
            else:
                print(f"❌ {variant_name}: FAILED (NaN loss)")
                
        except Exception as e:
            print(f"❌ {variant_name} failed with error: {e}")
            continue

    # Save raw results
    with open(results_dir / "data" / "focused_results.json", 'w') as f:
        results_for_json = []
        for r in results:
            r_copy = r.copy()
            if 'training_history' in r_copy:
                del r_copy['training_history']
            results_for_json.append(r_copy)
        json.dump(results_for_json, f, indent=2)

    # Create comprehensive visualizations
    create_plots(results, results_dir)

    # Generate HTML report
    generate_html_report(results, config, results_dir)

    # Console summary
    valid_results = [r for r in results if not math.isnan(r.get('val_loss', float('nan')))]
    
    print(f"\n📊 FOCUSED MUON ABLATION RESULTS:")
    print("=" * 100)
    print(f"{'Rank':<4} {'Variant':<35} {'Val Loss':<10} {'Val PPL':<10} {'Val Acc':<10} {'Time(s)':<8} {'Status':<8}")
    print("-" * 100)

    all_results_sorted = sorted(results, key=lambda x: x.get('val_loss', float('inf')))
    
    for i, result in enumerate(all_results_sorted):
        status = "✅" if not math.isnan(result.get('val_loss', float('nan'))) else "❌"
        val_loss_str = f"{result['val_loss']:.4f}" if not math.isnan(result.get('val_loss', float('nan'))) else "NaN"
        val_ppl_str = f"{result['val_perplexity']:.2f}" if not math.isnan(result.get('val_perplexity', float('nan'))) else "NaN"
        
        print(f"{i+1:<4} {result['variant']:<35} {val_loss_str:<10} {val_ppl_str:<10} "
              f"{result['val_accuracy']:<10.4f} {result['training_time']:<8.1f} {status}")

    # Final insights
    if valid_results:
        best = min(valid_results, key=lambda x: x['val_loss'])
        fastest = min(valid_results, key=lambda x: x['training_time'])
        
        print(f"\n🏆 FINAL INSIGHTS:")
        print(f"   🥇 Best Performer: {best['variant']} (Val Loss: {best['val_loss']:.4f})")
        print(f"   ⚡ Speed Champion: {fastest['variant']} ({fastest['training_time']:.1f}s)")
        print(f"   📈 Success Rate: {len(valid_results)}/{len(results)} ({len(valid_results)/len(results)*100:.1f}%)")
        print(f"   📁 All results: {results_dir}")
        print(f"   🌐 Report: {results_dir / 'reports' / 'focused_ablation_report.html'}")
        
        