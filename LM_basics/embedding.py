import torch
from torch import nn, Tensor
from jaxtyping import Int, Float


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        num_embeddings: size of the vocabulary;
        embedding_dim: dimension of the embedding vectors, i.e., d_model
        """
        super(Embedding, self).__init__()
        self.embed_matrix_init = torch.empty(num_embeddings, embedding_dim,
                                             device=device, dtype=dtype)
        nn.init.trunc_normal_(self.embed_matrix_init, mean=0, std=1, a=-3, b=3)
        self.embed_matrix = nn.Parameter(self.embed_matrix_init)

    def forward(self, token_ids: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length d_model"]:
        """
        Lookup the embedding vectors for the given token IDs
        """
        return self.embed_matrix[token_ids]
    

def main():
    model = Embedding(4, 2, device=torch.device("cpu"), dtype=torch.float32)
    token_ids = torch.tensor([[0, 2],[1, 3]], device=torch.device("cpu"))
    print(model.state_dict())
    print(model.forward(token_ids))


if __name__ == "__main__":
    main()
