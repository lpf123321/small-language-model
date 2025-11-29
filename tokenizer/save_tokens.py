import numpy as np
import multiprocessing as mp
from typing import List
from tokenizer.fast_tokenizer import Tokenizer


def _encode_chunk(args) -> np.ndarray:
    """子进程函数：加载tokenizer并处理一个文本块"""
    vocab_path, merges_path, special_tokens, lines = args
    tokenizer = Tokenizer(vocab_path, merges_path, special_tokens)
    token_ids = [tid for tid in tokenizer.encode_iterable(lines)]
    return np.array(token_ids, dtype=np.uint16)


def parallel_encode_to_file(
    filepath: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    special_tokens: list[str] | None = None,
    num_workers: int | None = None,
    chunk_size: int = 5000,
):
    """
    并行编码整个文本文件并以 uint16 numpy array 形式保存。
    文件格式：NumPy 原生 .npy 文件
    """
    if num_workers is None:
        num_workers = max(mp.cpu_count() - 1, 1)

    with open(filepath, "r", encoding="utf-8") as f:
        chunks = []
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) >= chunk_size:
                chunks.append(buf)
                buf = []
        if buf:
            chunks.append(buf)

    args_list = [
        (vocab_path, merges_path, special_tokens, chunk)
        for chunk in chunks
    ]

    print(f"Encoding with {num_workers} workers, total chunks: {len(args_list)} ...")

    with mp.Pool(num_workers) as pool:
        arrays: List[np.ndarray] = pool.map(_encode_chunk, args_list)

    all_tokens = np.concatenate(arrays).astype(np.uint16)
    np.save(output_path, all_tokens)
    print(f"Saved {len(all_tokens)} tokens to {output_path} ({all_tokens.nbytes / 1024 / 1024:.2f} MB)")

    return all_tokens


def test_parallel_save():
    import time
    start = time.time()
    arr = parallel_encode_to_file(
        filepath="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_path="tokenizer/vocab.pkl",
        merges_path="tokenizer/merges.pkl",
        output_path="tokens_5M_uint16.npy",
        special_tokens=['<|endoftext|>'],
        num_workers=4,
        chunk_size=10000,
    )
    print(f"Total time: {time.time() - start:.2f}s")
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")


if __name__ == "__main__":
    test_parallel_save()
