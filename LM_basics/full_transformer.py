import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from einops import repeat
from LM_basics.pre_norm_block import RMSNorm, Multihead_self_attention_with_rope, SwiGLU, RotaryPositionalEmbedding
from LM_basics.embedding import Embedding
from LM_basics.linear import Linear

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        """
        d_model: Dimensionality of the Transformer block inputs.
        num_heads: Number of heads to use in multi-head self-attention.
        d_ff: Dimensionality of the position-wise feed-forward inner layer.
        d_ff: Dimensionality of the feed-forward inner layer.
        max_seq_len: Maximum sequence length to pre-cache.
        """
        super(Transformer_block, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.RMSNorm1 = RMSNorm(d_model=d_model)
        self.RMSNorm2 = RMSNorm(d_model=d_model)
        self.RoPE = RotaryPositionalEmbedding(theta=theta, d_k=d_model//num_heads, max_seq_len=max_seq_len)
        self.MHA = Multihead_self_attention_with_rope(d_model=self.d_model, num_heads=self.num_heads, 
                                               max_seq_len=self.max_seq_len, theta=self.theta, RoPE=self.RoPE)
        self.FFN = SwiGLU(d_model=self.d_model, d_ff=self.d_ff)

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        batch_size, seq_len, _ = x.shape
        # Handle truncation
        if seq_len > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            effective_seq_len = self.max_seq_len
        else:
            effective_seq_len = seq_len
            
        token_positions = repeat(
            torch.arange(effective_seq_len, device=x.device),
            "seq_len -> batch_size seq_len",
            batch_size = batch_size
        )
        y = x + self.MHA(self.RMSNorm1(x), token_positions)
        return y + self.FFN(self.RMSNorm2(y))
    

class Transformer_LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super(Transformer_LM, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            Transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                            max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers)
        ])
        self.RMSNorm = RMSNorm(d_model=d_model)
        self.output_embed = Linear(in_features=d_model, out_features=vocab_size)
        # self.output_embed.weights = self.embedding.embed_matrix

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.RMSNorm(x)
        return self.output_embed(x)
    