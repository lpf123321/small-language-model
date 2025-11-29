from typing import Iterable, Iterator
from functools import lru_cache
import regex
import pickle as pkl
import multiprocessing as mp
import numpy as np

"""Experience on accelerate a tokenizer:
1. Cach single-byte bytes object and pair merge results that occur frequently. 
    It prevents doing repetitive jobs.
2. When merging a pair of bytes object, do not create a new list. Use list.pop() instead.
3. Construct a dict that maps a pair in `merges` to its index in it. So the priority of a pair in a word
    can be found quickly."""
class Tokenizer:
    def __init__(self, vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            self.id_to_tokens = pkl.load(f)
        with open(merges_filepath, 'rb') as f:
            self.merges = pkl.load(f)
        self.tokens_to_id = {v: k for k, v in self.id_to_tokens.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)} # find the priority of a pair in O(1) time (important!)
        self.special_tokens = special_tokens if special_tokens is not None else []
        # contractions, letters, numbers, punctuation, single spaces, multiple spaces
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pattern = regex.compile(self.PAT)
        self._bytes_pool = {i: bytes([i]) for i in range(256)} # cach single-byte bytes object
        self._special_token_bytes = {} # pre-calc the bytes repre of special tokens 
        for sp_token in self.special_tokens:
            tok_bytes = sp_token.encode('utf-8')
            self._special_token_bytes[sp_token] = tok_bytes
            if tok_bytes not in self.tokens_to_id:
                self.tokens_to_id[tok_bytes] = len(self.tokens_to_id)
                self.id_to_tokens[self.tokens_to_id[tok_bytes]] = tok_bytes

    @lru_cache(maxsize=100_000)
    def _merge_cached(self, word: str) -> tuple:
        """cach pair merge results that occur frequently"""
        bytes_list: list[bytes] = [self._bytes_pool[b] for b in word.encode('utf-8')]
        return tuple(self._merge_bytes(bytes_list))

    def _merge_bytes(self, bytes_list: list[bytes]) -> list[bytes]:
        """merge bytes in a word (splitted into a bytes list) from pre-tokenization"""
        while True:
            # find prior pairs to merge
            best_pair = None
            best_pos = -1
            min_rank = float('inf')
            
            for i in range(len(bytes_list) - 1):   
                pair = (bytes_list[i], bytes_list[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair
                        best_pos = i
            
            if best_pair is None:
                break   
            # merge
            bytes_list[best_pos] = best_pair[0] + best_pair[1]  # merged to the first position    
            bytes_list.pop(best_pos+1)  # remove the second position (do not create a new list!)
        
        return bytes_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files."""
        for line in iterable:
            # assume that iterable is a file object
                # assume that when a special token occurs, it takes a whole line
            if line.startswith('<|endoftext|>'):
                yield self.tokens_to_id[b'<|endoftext|>']
                yield self.tokens_to_id[b'\n']
            else:
                words: list[str] = self.pattern.findall(line)
                for word in words:
                    # bytes_list: list[bytes] = [self._bytes_pool[b] for b in word.encode('utf-8')] 
                    # # list of single-byte bytes 
                    # if len(bytes_list) == 0:
                    #     continue
                    # bytes_list: list[bytes] = self._merge_bytes(bytes_list)
                    for t in self._merge_cached(word):
                        yield self.tokens_to_id[t]

        
def test():
    import time
    time1 = time.time()
    tokenizer = Tokenizer(
        'tokenizer/vocab.pkl', 
        'tokenizer/merges.pkl', 
        special_tokens=['<|endoftext|>']
    )
    with open("tests/fixtures/tinystories_sample_5M.txt") as f:
        encoder = tokenizer.encode_iterable(f)
        i = 0
        for token in encoder:
            i += 1
    time2 = time.time()
    print(f"total time: {time2-time1}s.")

# ======== 并行化改进部分 ========

def _init_worker(vocab_path, merges_path, special_tokens):
    """每个子进程初始化一次 tokenizer"""
    global TOKENIZER
    TOKENIZER = Tokenizer(vocab_path, merges_path, special_tokens)


def _encode_chunk(lines: list[str]) -> np.ndarray:
    """子进程任务：编码文本块为 np.uint16 数组"""
    global TOKENIZER
    token_ids = np.fromiter(TOKENIZER.encode_iterable(lines), dtype=np.uint16)
    return token_ids


def parallel_encode_file_to_uint16(
    filepath: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    special_tokens: list[str] | None = None,
    num_workers: int | None = None,
    chunk_size: int = 5000,
):
    """
    并行编码整个文本文件并将结果流式写入二进制文件 (uint16)
    """
    if num_workers is None:
        num_workers = max(mp.cpu_count() - 1, 1)
    print(f"Using {num_workers} workers, chunk size = {chunk_size} lines")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(vocab_path, merges_path, special_tokens)) as pool, \
         open(filepath, "r", encoding="utf-8") as fin, \
         open(output_path, "wb") as fout:

        # 按块生成器：避免一次性加载整个文件
        def chunk_generator():
            buf = []
            for line in fin:
                buf.append(line)
                if len(buf) >= chunk_size:
                    yield buf
                    buf = []
            if buf:
                yield buf

        # 异步处理每个chunk并流式写出
        total_tokens = 0
        for arr in pool.imap(_encode_chunk, chunk_generator()):
            arr.tofile(fout)
            total_tokens += len(arr)
        print(f"✅ Finished. Total tokens: {total_tokens}")


# ======== test entrance ========

def save_parallel():
    import time
    start = time.time()
    parallel_encode_file_to_uint16(
        filepath="data/TinyStoriesV2-GPT4-train.txt",
        vocab_path="tokenizer/vocab.pkl",
        merges_path="tokenizer/merges.pkl",
        output_path="data/tokens_train.bin",
        special_tokens=['<|endoftext|>'],
        num_workers=4,
        chunk_size=10000,
    )
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    test()