import torch
from torch import nn
from math import sqrt


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(Linear, self).__init__()
        self.std = sqrt(2 / (in_features + out_features))
        self.weights_init = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(self.weights_init, mean=0, std=self.std, 
                              a=-3*self.std, b=3*self.std)
        self.weights = nn.Parameter(self.weights_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return torch.matmul(x, self.weights.T)
        # return einsum(
        #     self.weights, x,
        #     "out_features in_features, ... in_features -> ... out_features"
        # )  # we do not include a bias term, following most modern LLMs
    

def main():
    model = Linear(in_features=3, out_features=3, 
                   device=torch.device("cpu"), dtype=torch.float32)
    print(model.state_dict())


if __name__ == "__main__":
    main()
