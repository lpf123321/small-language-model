import torch
import numpy as np
import numpy.typing as npt
from torch import Tensor
import os
import typing


def data_loader(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    This function randomly samples starting positions for each sequence in the batch.
    """
    assert dataset.ndim == 1, "Dataset must be a 1D numpy array of token IDs"
    assert len(dataset) > context_length, f"Dataset length {len(dataset)} must be greater than context_length {context_length}"
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    x = np.stack([
        dataset[start_idx:start_idx + context_length]
        for start_idx in start_indices
    ])
    # Labels are the next tokens (shifted by 1)
    y = np.stack([
        dataset[start_idx + 1:start_idx + context_length + 1]
        for start_idx in start_indices
    ])
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    """
    Dump all the state from the first three parameters into the file-like object `out`.
    """
    obj: dict = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration}
    torch.save(obj=obj, f=out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load a checkpoint from `src` (path or file-like object), 
    recover the model and optimizer states from that checkpoint.
    Return the iteration number that was saved to the checkpoint.
    """
    obj: dict = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


def test():
    dataset = np.arange(20)
    x, y = data_loader(dataset, batch_size=3, context_length=5, device="cpu")
    print("x:\n", x)
    print("y:\n", y)


if __name__ == "__main__":
    test()