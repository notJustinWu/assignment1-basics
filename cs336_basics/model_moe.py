import torch
import torch.nn as nn
import logging
import numpy as np
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(
            self.out_features, self.in_features, device=device, dtype=dtype))
        std = (2/(in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean = 0.0, std = std, a = -3*std, b = 3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.empty(
            self.num_embeddings, self.embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, mean = 0.0, std = 0.05, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.gain = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)        
        rms = (((x**2).mean(dim=-1, keepdim=True)) + self.eps) ** 0.5
        x_norm = x / rms 
        result = x_norm * self.gain
        return result.to(in_dtype)

class positionwise_feedforward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear3 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.silu(self.linear1(x)) * self.linear3(x)
        return self.linear2(gated)

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.positions = torch.arange(
            max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        self.block_indices = torch.arange(
            d_k // 2, dtype=torch.float32, device=device).unsqueeze(0)
        self.exponents = (2 * self.block_indices)/d_k
        self.inv_freq = 1.0/(theta ** self.exponents)
        self.angles = self.positions * self.inv_freq
        
        sin_buffer = torch.sin(self.angles)
        cos_buffer = torch.cos(self.angles)
        
        self.register_buffer("sin_buffer", sin_buffer, persistent=False)
        self.register_buffer("cos_buffer", cos_buffer, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin_buffer[token_positions, :]
        cos = self.cos_buffer[token_positions, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_odd * cos + x_even * sin

        x_out = torch.zeros_like(x)
        x_out[..., 0::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out

def softmax(x: torch.Tensor, dim: int):
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    return exp_x/torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    pre_softmax = QK/(d_k ** 0.5)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    attn_weights = softmax(pre_softmax, dim=-1)
    attn = einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    return attn

class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads 
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads
        self.W_Q = Linear(d_model, num_heads * self.dk).W
        self.W_K = Linear(d_model, num_heads * self.dk).W
        self.W_V = Linear(d_model, num_heads * self.dv).W
        self.W_O = Linear(num_heads * self.dv, d_model).W
        self.qk_norm = RMSNorm(d_model=d_model, eps=1e-5)

    def forward(self, x: torch.Tensor, positions=None, theta=None, max_seq_len=None):

        seq_len = x.shape[-2]

        # project x to Q, K, V
        Q = einsum(self.W_Q, x, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
        K = einsum(self.W_K, x, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
        V = einsum(self.W_V, x, "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v")

        Q = self.qk_norm(Q)
        K = self.qk_norm(K)

        Q = rearrange(Q, "... seq_len (head d) -> ... head seq_len d", head=self.num_heads)
        K = rearrange(K, "... seq_len (head d) -> ... head seq_len d", head=self.num_heads)
        V = rearrange(V, "... seq_len (head d) -> ... head seq_len d", head=self.num_heads)

        if positions is not None:
            ROPE = RotaryPositionalEmbedding(theta=theta, d_k=self.dk, max_seq_len=max_seq_len, device=x.device)
            Q = ROPE(Q, positions)
            K = ROPE(K, positions)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()

        multi_attn = scaled_dot_product_attention(Q, K, V, mask=~causal_mask)
        concat_attn = rearrange(multi_attn, "... head seq_len d_v -> ... seq_len (head d_v)")
        result = einsum(self.W_O, concat_attn, "d_model hdv, ... seq_len hdv -> ... seq_len d_model")

        return result

class transformer_block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm1 = RMSNorm(d_model=self.d_model, eps=1e-5)
        self.norm2 = RMSNorm(d_model=self.d_model, eps=1e-5)
        self.attn = multihead_self_attention(d_model=self.d_model, num_heads=num_heads)
        self.ffn = positionwise_feedforward(d_model=self.d_model, d_ff=self.d_ff)
    def forward(self, x: torch.Tensor, positions=None, theta=None, max_seq_len=None):
        y = x + self.attn(self.norm1(x), positions=positions, theta=theta, max_seq_len=max_seq_len)
        z = y + self.ffn(self.norm2(y))

        return z
    
class transformer_lm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(self.num_layers)])
        self.norm = RMSNorm(d_model=d_model, eps=1e-5)
        #self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices, positions=None, theta=None):
        x = self.token_embedding(in_indices)
        
        for block in self.transformer_blocks:
            x = block(x, positions=positions, theta=theta, max_seq_len=self.context_length)
            
        result = einsum(self.token_embedding.embedding, self.norm(x), "num_embeddings embedding_dim, ... embedding_dim -> ... num_embeddings")

        return result

def cross_entropy(logits, targets):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_stable = logits - logits_max
    exp_logits = torch.exp(logits_stable)
    denom = torch.sum(exp_logits, dim=-1)
    indices = [torch.arange(logits_stable.size(0)), targets]
    neg_log_prob = torch.log(denom)-logits_stable[indices]
    # neg_log_prob = torch.log(denom)-logits_stable[torch.arange(logits_stable.size(0)), targets]
    return torch.mean(neg_log_prob)

def cross_entropy_z_loss(logits, targets, alpha=1e-2):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_stable = logits - logits_max
    exp_logits = torch.exp(logits_stable)
    denom = torch.sum(exp_logits, dim=-1)
    indices = [torch.arange(logits_stable.size(0)), targets]
    neg_log_prob = torch.log(denom)-logits_stable[indices]

    log_z = torch.logsumexp(logits, dim=-1)           # (B,)
    z_loss = (log_z**2).mean()
    # neg_log_prob = torch.log(denom)-logits_stable[torch.arange(logits_stable.size(0)), targets]
    return torch.mean(neg_log_prob) + alpha * z_loss

def softplus(x: torch.Tensor, beta = 1.0, threshold = 20.0):
    x_beta = beta * x
    exp_arg = torch.clamp(x_beta, max=threshold)
    soft = (1.0/beta) * torch.log(1 + torch.exp(exp_arg))
    return torch.where(x_beta > threshold, x, soft)

class TopkRouter(torch.nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        self.num_experts = num_experts
        self.topklinear = Linear(d_model, num_experts)
        self.noiselinear = Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        raw_logits = self.topklinear(x) # shape: (..., num_experts)
        raw_noise_logits = self.noiselinear(x)
        
        noise = torch.normal(torch.zeros(raw_noise_logits.shape), std=1)

        noise_tensor = softplus(raw_noise_logits) * noise

        combined_tensor = noise_tensor + raw_logits

        top_k_values, top_k_indices = torch.topk(combined_tensor, self.top_k, dim = -1)
        zeros = torch.full_like(raw_logits, float('-inf'))
        new_logits = zeros.scatter(-1, top_k_indices, top_k_values)

        return softmax(new_logits, dim=-1), top_k_indices

class SparseMoE(torch.nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.router = TopkRouter(d_model, num_experts, top_k)
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList([
            positionwise_feedforward(d_model, d_ff = int(8/3 * d_model))
            for _ in range(self.num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):

        # for each batch and word in that batch, have a list of best experts and their weights
        gating_values, top_k_indices = self.router(x)
        ans = torch.zeros_like(x)

        B, S, K = top_k_indices.shape
        
        for i, expert in enumerate(self.experts):
            expert_picks = (top_k_indices == i)
            if not expert_picks.any():
                continue

            # this gives the tensors for the (batch, indices) for which the expert is active
            batch_idx, seq_idx = expert_picks.any(dim=-1).nonzero(as_tuple=True)

            transformed = expert(x[batch_idx, seq_idx]) # (..., d_model)
            gating = rearrange(gating_values[batch_idx, seq_idx, i], 'p -> p 1')
            transformed = transformed * gating
            ans[batch_idx, seq_idx] += transformed
        
        return ans

class transformer_block_with_MoE(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.top_k = top_k
        self.num_experts = num_experts
        self.d_ff = d_ff
        self.norm1 = RMSNorm(d_model=self.d_model, eps=1e-5)
        self.norm2 = RMSNorm(d_model=self.d_model, eps=1e-5)
        self.attn = multihead_self_attention(d_model=self.d_model, num_heads=num_heads)
        self.MoE = SparseMoE(d_model=self.d_model, num_experts=self.num_experts, top_k=self.top_k)
    def forward(self, x: torch.Tensor, positions=None, theta=None, max_seq_len=None):
        y = x + self.attn(self.norm1(x), positions=positions, theta=theta, max_seq_len=max_seq_len)
        z = y + self.MoE(self.norm2(y))

        return z

class transformer_lm_with_MoE(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            transformer_block_with_MoE(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_experts=self.num_experts, top_k=self.top_k
            ) 
            for _ in range(self.num_layers)
        ])
        self.norm = RMSNorm(d_model=d_model, eps=1e-5)

    def forward(self, in_indices, positions=None, theta=None):
        x = self.token_embedding(in_indices)
        
        for block in self.transformer_blocks:
            x = block(x, positions=positions, theta=theta, max_seq_len=self.context_length)
            
        result = einsum(self.token_embedding.embedding, self.norm(x), "num_embeddings embedding_dim, ... embedding_dim -> ... num_embeddings")

        return result