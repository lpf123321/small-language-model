from typing import Iterable, Iterator
import regex
import pickle as pkl
from functools import lru_cache


class Tokenizer(object):
    """A Byte-Pair Encoding (BPE) tokenizer."""

    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        """Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens."""
        self.id_to_tokens = vocab
        self.tokens_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)} # find the priority of a pair in O(1) time
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
            bytes_list.pop(best_pos+1)  # remove the second position
        
        return bytes_list


    @classmethod
    def from_files(cls, vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        """Class method that constructs and returns a Tokenizer 
        from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and 
        (optionally) a list of special tokens."""
        with open(vocab_filepath, 'rb') as f:
            vocab = pkl.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pkl.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""     
        ## 1. pre-tokenize the text into tokens (using self.PAT)
        ##    - make sure to handle special tokens (if any) correctly
        ##      (i.e., special tokens should be treated as indivisible tokens)
        ids: list[int] = []
        special_tokens = []
        if self.special_tokens:
            special_tokens = [sp for sp in sorted(self.special_tokens, key=len, reverse=True)]
        i = 0
        while i < len(text):
            for sp_token in special_tokens: # Longest match first
                if text.startswith(sp_token, i):
                    word_bytes: bytes = self._special_token_bytes[sp_token]
                    ids.append(self.tokens_to_id[word_bytes])
                    i += len(sp_token)
                    break
              # normal text segment
            j = i
            while j <= len(text): # find the next special token or end of text
                if self.special_tokens and any(text.startswith(sp_token, j) for sp_token in self.special_tokens):
                    break # stop at the start of the next special token
                j += 1
            words: list[str] = self.pattern.findall(text[i:j])
            for word in words:
                bytes_list: list[bytes] = [bytes([b]) for b in word.encode('utf-8')] # list of single-byte
                ## 2. apply BPE merges to each token (if possible)
                if len(bytes_list) == 0:
                    continue
                bytes_list = self._merge_bytes(bytes_list)
                ## 3. map each token to its ID using self.tokens_to_id
                for w in bytes_list:
                    ids.append(self.tokens_to_id[w])
            i = j
        ## 4. return the list of token IDs
        return ids

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files."""
        for line in iterable:
            # assume that iterable is a file object
            # for sp_token in self.special_tokens:
                # assume that when a special token occurs, it takes a whole line
            if line.startswith('<|endoftext|>'):
                yield self.tokens_to_id[b'<|endoftext|>']
                yield self.tokens_to_id[b'\n']
            else:
                words: list[str] = self.pattern.findall(line)
                for word in words:
                    for t in self._merge_cached(word):
                        yield self.tokens_to_id[t]

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        token_bytes: list[bytes] = []
        for id_ in ids:
            if id_ in self.id_to_tokens:
                token_bytes.append(self.id_to_tokens[id_])
            else:
                raise KeyError(f"Token ID {id_} not found in vocabulary.")
        return b''.join(token_bytes).decode('utf-8', errors='replace')


def test():
    import time
    text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # text = "Once upon a time there was a little boy named Ben."
    tokenizer = Tokenizer.from_files(
        'tokenizer/vocab.pkl', 
        'tokenizer/merges.pkl', 
        special_tokens=['<|endoftext|>']
    )
    time1 = time.time()
    ids: list[int] = tokenizer.encode(text)
    print(ids)
    result: str = tokenizer.decode(ids)
    print(result)
    print(result == text)  # should be True
    time2 = time.time()
    print(f"total time: {time2-time1}s")

def test2():
    import time
    time1 = time.time()
    tokenizer = Tokenizer.from_files(
        'tokenizer/vocab.pkl', 
        'tokenizer/merges.pkl', 
        special_tokens=['<|endoftext|>']
    )
    with open("tests/fixtures/tinystories_sample_5M.txt") as f:
        encoder = tokenizer.encode_iterable(f)
        i = 0
        for token in encoder:
            i += 1
        print(i)
    time2 = time.time()
    print(f"total time: {time2-time1}s.")

if __name__ == '__main__':
    test2()
