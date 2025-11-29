import torch
from torch import nn
from LM_basics.linear import Linear
from jaxtyping import Bool, Float, Int
from torch import Tensor
from math import pow, sin, cos, sqrt
from einops import rearrange


def linear_init(in_features: int, out_features: int) -> Tensor:
    std = sqrt(2 / (in_features + out_features))
    weights_init = torch.empty(out_features, in_features)
    nn.init.trunc_normal_(weights_init, mean=0, std=std, 
                                a=-3*std, b=3*std)
    return weights_init


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape
        """
        in_dtype = x.dtype
        # upcast input to torch.float32 to prevent overflow when squaring the input
        x = x.to(torch.float32)
        result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weights
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(SwiGLU, self).__init__()
        self.linear1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        temp = self.linear1(x)
        return self.linear2(
            temp * torch.sigmoid(temp) * 
            self.linear3(x)
        )



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                device: torch.device | None = None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.matrices = torch.zeros(max_seq_len, d_k, d_k, device=device)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                theta_ik = i / pow(theta, (2*k)/d_k)
                self.matrices[i, 2*k, 2*k] = cos(theta_ik)
                self.matrices[i, 2*k, 2*k+1] = -sin(theta_ik)
                self.matrices[i, 2*k+1, 2*k] = sin(theta_ik)
                self.matrices[i, 2*k+1, 2*k+1] = cos(theta_ik)
        self.register_buffer("rope_matrices", self.matrices, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and 
        return a tensor of the same shape. The token positions (..., seq_len) 
        specify the token positions of x along the sequence dimension.
        """
        # Clamp token positions to be within valid range
        max_pos = self.rope_matrices.shape[0] - 1  # type: ignore
        token_positions = torch.clamp(token_positions, max=max_pos)
        matrices_select = self.rope_matrices[token_positions] # type: ignore
        # Handle multi-head attention case with more memory-efficient approach
        # x shape: (batch_size, num_heads, seq_len, d_k)
        # matrices_select shape: (batch_size, seq_len, d_k, d_k)
        if x.dim() == 4:  # Multi-head case
            # Instead of creating a huge 5D tensor, use einsum for more efficient computation, avoiding the memory-intensive unsqueeze operation
            return torch.einsum('bsij,bhsj->bhsi', matrices_select, x)
        else:
            return torch.matmul(matrices_select, x.unsqueeze(-1)).squeeze(-1)
         


def Softmax(in_features: Tensor, dim: int):
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.
    """
    in_features -= in_features.max(dim=dim, keepdim=True)[0]
    in_features = torch.exp(in_features)
    return in_features / in_features.sum(dim=1, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"], # values == keys
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k: int = Q.shape[-1]
    att_score: Tensor = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    # einsum(
    #     Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    # ) / sqrt(d_k)
    if mask is not None:
        att_score = att_score.masked_fill(mask=~mask, value=float('-inf'))
    weights: Tensor = torch.softmax(att_score, dim=-1)
    return torch.matmul(weights, V)
    # return einsum(
    #     weights, V, "... queries keys, ... keys d_v -> ... queries d_v"
    # )

    
class Multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        d_model: Dimensionality of the Transformer block inputs.
        num_heads: Number of heads to use in multi-head self-attention
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.W_Q = nn.Parameter(linear_init(d_model, d_model))
        self.W_K = nn.Parameter(linear_init(d_model, d_model))
        self.W_V = nn.Parameter(linear_init(d_model, d_model))
        self.W_O = nn.Parameter(linear_init(d_model, d_model))

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, " ... seq_len d_model"]:
        Q = rearrange(torch.matmul(x, self.W_Q.T),
                      "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
                      num_heads = self.num_heads)
        K = rearrange(torch.matmul(x, self.W_K.T),
                      "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
                      num_heads = self.num_heads)
        V = rearrange(torch.matmul(x, self.W_V.T),
                      "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
                      num_heads = self.num_heads)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        heads: Tensor = scaled_dot_product_attention(Q, K, V, mask)
        concat = rearrange(
            heads, 
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )
        return torch.matmul(concat, self.W_O.T)


class Multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, RoPE: nn.Module):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.RoPE = RoPE
        self.W_Q = nn.Parameter(linear_init(d_model, d_model))
        self.W_K = nn.Parameter(linear_init(d_model, d_model))
        self.W_V = nn.Parameter(linear_init(d_model, d_model))
        self.W_O = nn.Parameter(linear_init(d_model, d_model))

    def forward(self, 
                x: Float[Tensor, "... seq_len d_model"],
                token_positions: Int[Tensor, " ... seq_len"] | None = None) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.shape[-2]
        Q = rearrange(torch.matmul(x, self.W_Q.T),
                      "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
                      num_heads = self.num_heads)
        K = rearrange(torch.matmul(x, self.W_K.T),
                      "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
                      num_heads = self.num_heads)
        if token_positions is not None:
            Q = self.RoPE(Q, token_positions)
            K = self.RoPE(K, token_positions)
        V = rearrange(torch.matmul(x, self.W_V.T),
                      "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
                      num_heads = self.num_heads)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).unsqueeze(0).unsqueeze(0)
        heads: Tensor = scaled_dot_product_attention(Q, K, V, mask)
        concat = rearrange(
            heads, 
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )
        return torch.matmul(concat, self.W_O.T)
        