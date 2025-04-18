import torch
import torch.nn as nn
import logging
import numpy as np
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    """
    Linear layer with truncated normal initialization
    Args:
        in_features: int
        out_features: int
        device: torch.device
        dtype: torch.dtype
    Returns:
        torch.Tensor: The output tensor of shape (..., out_features)
    """
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
    """
    Embedding layer with truncated normal initialization
    Args:
        num_embeddings: int
        embedding_dim: int
        device: torch.device
        dtype: torch.dtype
    Returns:
        torch.Tensor: The output tensor of shape (..., embedding_dim)
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.empty(
            self.num_embeddings, self.embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, mean = 0.0, std = 1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    
class RMSNorm(torch.nn.Module):
    """
    RMSNorm layer
    Args:
        d_model: int
        eps: float
        device: torch.device
        dtype: torch.dtype
    Returns:
        torch.Tensor: The output tensor of shape (..., d_model)
    """
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
    """
    Positionwise feedforward layer
    Args:
        d_model: int
        d_ff: int
    Returns:
        torch.Tensor: The output tensor of shape (..., d_model)
    """
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
    """
    Rotary positional embedding layer
    Args:
        theta: float
        d_k: int
        max_seq_len: int
        device: torch.device
        dtype: torch.dtype
    """
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
    """
    Softmax function
    Args:
        x: torch.Tensor
        dim: int
    """
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    return exp_x/torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    """
    Scaled dot product attention
    Args:
        Q: torch.Tensor
        K: torch.Tensor
        V: torch.Tensor
        mask: torch.Tensor
    """
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    pre_softmax = QK/(d_k ** 0.5)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    attn_weights = softmax(pre_softmax, dim=-1)
    attn = einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    return attn

class multihead_self_attention(torch.nn.Module):
    """
    Multihead self-attention layer
    Args:
        d_model: int
        num_heads: int
    """
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

    def forward(self, x: torch.Tensor, positions=None, theta=None, max_seq_len=None):

        seq_len = x.shape[-2]

        # project x to Q, K, V
        Q = einsum(self.W_Q, x, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
        K = einsum(self.W_K, x, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
        V = einsum(self.W_V, x, "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v")

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
    """
    Transformer block layer
    Args:
        d_model: int
        num_heads: int
        d_ff: int
    """
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
    """
    Transformer language model
    Args:
        vocab_size: int
        context_length: int
        num_layers: int
        d_model: int
        num_heads: int
        d_ff: int
    """
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(self.num_layers)])
        self.norm = RMSNorm(d_model=d_model, eps=1e-5)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices, positions=None, theta=None):
        x = self.token_embedding(in_indices)
        
        for block in self.transformer_blocks:
            x = block(x, positions=positions, theta=theta, max_seq_len=self.context_length)
            
        result = self.linear(self.norm(x))

        return result

def cross_entropy(logits, targets):
    """
    Cross entropy loss
    Args:
        logits: torch.Tensor
        targets: torch.Tensor
    """
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_stable = logits - logits_max
    exp_logits = torch.exp(logits_stable)
    denom = torch.sum(exp_logits, dim=-1)
    indices = [torch.arange(logits_stable.size(0)), targets]
    neg_log_prob = torch.log(denom)-logits_stable[indices]
    # neg_log_prob = torch.log(denom)-logits_stable[torch.arange(logits_stable.size(0)), targets]
    return torch.mean(neg_log_prob)

########################################################
# ABLATIONS ############################################
########################################################

# ABLATION 1: Remove RMSNorm ##########################

class transformer_block_ablation_no_rms_norm(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = multihead_self_attention(d_model=self.d_model, num_heads=num_heads)
        self.ffn = positionwise_feedforward(d_model=self.d_model, d_ff=self.d_ff)
    def forward(self, x: torch.Tensor, positions=None, theta=None, max_seq_len=None):
        y = x + self.attn(x, positions=positions, theta=theta, max_seq_len=max_seq_len)
        z = y + self.ffn(y)

        return z

class transformer_lm_ablation_no_rms_norm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            transformer_block_ablation_no_rms_norm(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(self.num_layers)])
        self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices, positions=None, theta=None):
        x = self.token_embedding(in_indices)
        
        for block in self.transformer_blocks:
            x = block(x, positions=positions, theta=theta, max_seq_len=self.context_length)
            
        result = self.linear(x)

        return result

# ABLATION 2: Post RMSNorm ##########################

class transformer_block_ablation_post_rms_norm (torch.nn.Module):
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
        y = self.norm1(x + self.attn(x, positions=positions, theta=theta, max_seq_len=max_seq_len))
        z = self.norm2(y + self.ffn(y))

        return z

class transformer_lm_ablation_post_rms_norm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(self.num_layers)])
        self.norm = RMSNorm(d_model=d_model, eps=1e-5)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices, positions=None, theta=None):
        x = self.token_embedding(in_indices)
        
        for block in self.transformer_blocks:
            x = block(x, positions=positions, theta=theta, max_seq_len=self.context_length)
            
        result = self.linear(self.norm(x))

        return result

# ABLATION 3: No Position Embedding ##########################

# JUST DO THE SAME THING BUT PASS IN NONE FOR POSITIONS AND THETA

# ABLATION 4: Silu Activation Function
class positionwise_feedforward_ablation_silu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.silu(self.linear1(x)))

class transformer_block_ablation_silu(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm1 = RMSNorm(d_model=self.d_model, eps=1e-5)
        self.norm2 = RMSNorm(d_model=self.d_model, eps=1e-5)
        self.attn = multihead_self_attention(d_model=self.d_model, num_heads=num_heads)
        self.ffn = positionwise_feedforward_ablation_silu(d_model=self.d_model, d_ff=self.d_ff)
    def forward(self, x: torch.Tensor, positions=None, theta=None, max_seq_len=None):
        y = x + self.attn(self.norm1(x), positions=positions, theta=theta, max_seq_len=max_seq_len)
        z = y + self.ffn(self.norm2(y))

        return z

class transformer_lm_ablation_silu(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            transformer_block_ablation_silu(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(self.num_layers)])
        self.norm = RMSNorm(d_model=d_model, eps=1e-5)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices, positions=None, theta=None):
        x = self.token_embedding(in_indices)
        
        for block in self.transformer_blocks:
            x = block(x, positions=positions, theta=theta, max_seq_len=self.context_length)
            
        result = self.linear(self.norm(x))

        return result
