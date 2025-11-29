from utils.pretokenization_example import find_chunk_boundaries
import regex as re
import os
from collections import Counter
import heapq
import pickle as pkl
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List, Optional


@dataclass
class Word:
    symbols: list[int]   # 当前符号 ID 序列
    count: int           # 权重（预 token 出现次数）
    frozen: bool = False # 特殊 token（不在内部合并）

def _worker_pretokenize(args: Tuple[str, int, int, int, int, str, bytes, str, bool]):
    """Worker 进程：假设特殊 token 总是独占一行进行流式预分词，并可打印进度。

    args tuple:
        path: 文件路径
        start: 起始字节偏移（包含）
        end: 结束字节偏移（不包含）
        chunk_idx: 当前 chunk 序号
        total_chunks: 总 chunk 数
        pattern: 预分词正则
        special_token_bytes: 特殊 token 的 bytes
        special_token_str: 特殊 token 的字符串表示
        show_progress: 是否打印进度
    """
    (
        path,
        start,
        end,
        chunk_idx,
        total_chunks,
        pattern,
        special_token_bytes,
        special_token_str,
        show_progress,
    ) = args
    pat = re.compile(pattern)
    counts = Counter()
    with open(path, 'rb') as f:
        f.seek(start)
        current = start
        chunk_size = end - start
        # 期望打印 ~50 次以内
        report_every = max(1, min(50, chunk_size // (1024 * 1024 * 2)))  # 每 ~2MB 或更少次数
        bytes_since_report = 0
        reports = 0
        while current < end:
            remaining = end - current
            line = f.readline(remaining)
            if not line:
                break
            current += len(line)
            bytes_since_report += len(line)
            stripped = line.strip()
            if stripped == special_token_bytes:
                counts[special_token_str] += 1
            else:
                # 解码后使用 finditer 逐个更新，避免生成整个匹配列表
                text = line.decode('utf-8', errors='ignore')
                if text:
                    for m in pat.finditer(text):
                        tok = m.group(0)
                        counts[tok] += 1
            if show_progress and bytes_since_report and chunk_size > 0:
                # 粗略控制打印频率
                if bytes_since_report >= (chunk_size / 50) or reports < 3 and bytes_since_report >= (chunk_size / 200) or bytes_since_report >= 8 * 1024 * 1024:
                    pct = (current - start) / chunk_size * 100.0
                    print(f"[pretok] chunk {chunk_idx+1}/{total_chunks} {pct:5.1f}% ({current-start}/{chunk_size} bytes)", flush=True)
                    bytes_since_report = 0
                    reports += 1
    return counts


class Tokenizer_trainer(object):
    def __init__(self, file_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> None:
        self.path = str(file_path)
        self.vocab_size = vocab_size
        self.num_processes = 4
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # noqa: E501
        self.special_tokens = special_tokens[:]  # 特殊 token 列表
        self.token_counts: Counter[str] = Counter()  # 预分词 token 频次（字符串级）
        self.words: List[Word] = []
        # vocab 符号映射
        self.id_to_bytes: Dict[int, bytes] = {}
        self.bytes_to_id: Dict[bytes, int] = {}
        self.symbol_meta: Dict[int, tuple[int, int]] = {}  # 仅 merge 产生的符号记录 (left,right)
        self.next_symbol_id: int = 0
        # pair 统计
        self.pair_freq: Dict[tuple[int, int], int] = {}
        # pair_index[(a,b)] = { word_id: [pos1, pos2, ...], ... }
        self.pair_index: Dict[tuple[int, int], Dict[int, List[int]]] = {}
        self.merges: List[tuple[bytes, bytes]] = []  # merge 结果 (bytes, bytes)
        # 维护最高频 pair 的最小堆（存负频次实现最大堆效果）元素: (-freq, a, b)
        self.pair_heap: List[tuple[int, int, int]] = []


    # ------------------------ 预分词主流程 ------------------------ #
    def pretokenize_counts(self, show_progress: bool = True):
        """对整个语料进行预分词。

        步骤：
          1. 选取一个特殊 token 作为边界（若包含 <|endoftext|> 优先）
          2. 计算 chunk 边界
          3. 多进程映射 `_worker_pretokenize`
          4. 聚合结果
        """
        if not os.path.isfile(self.path):  # 基本健壮性检查
            raise FileNotFoundError(self.path)

        # 选择一个用于 boundary 的特殊 token
        boundary_token = None
        if '<|endoftext|>' in self.special_tokens:
            boundary_token = '<|endoftext|>'
        elif self.special_tokens:
            boundary_token = self.special_tokens[0]
        else:
            # 若没有特殊 token, 人为指定一个不会出现的分隔符（几乎不影响，只是 chunk 更粗糙）
            boundary_token = '\u0000\u0000<<NO_SPLIT_TOKEN>>\u0000'
        boundary_bytes = boundary_token.encode('utf-8')
        with open(self.path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, boundary_bytes)
        # 至少生成一个区间
        if len(boundaries) < 2:
            return Counter()
        # 构造 (start, end) pairs
        spans = list(zip(boundaries[:-1], boundaries[1:]))
        # 根据实际 chunk 数修正进程数，避免浪费
        num_workers = min(self.num_processes, len(spans)) or 1
        total_chunks = len(spans)
        worker_args = [
            (self.path, start, end, idx, total_chunks, self.PAT, boundary_bytes, boundary_token, show_progress)
            for idx, (start, end) in enumerate(spans)
        ]

        if num_workers == 1:
            # 退化为串行（方便调试）
            counters = [_worker_pretokenize(a) for a in worker_args]
        else:
            with Pool(processes=num_workers) as pool:
                counters = pool.map(_worker_pretokenize, worker_args)

        total = Counter()
        for c in counters:
            total.update(c)
        self.token_counts = total

    def iter_pretokens(self) -> Iterable[str]:
        """单进程惰性迭代。假设特殊 token 独占一行。"""
        pat = re.compile(self.PAT)
        special = '<|endoftext|>' if '<|endoftext|>' in self.special_tokens else (self.special_tokens[0] if self.special_tokens else None)
        special_bytes = special.encode('utf-8') if special else None
        with open(self.path, 'rb') as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if special_bytes and stripped == special_bytes:
                    yield special  # type: ignore[arg-type]
                else:
                    text = raw_line.decode('utf-8', errors='ignore')
                    if text:
                        for t in pat.findall(text):
                            yield t

    # ------------------------ BPE 初始化阶段 ------------------------ #
    def build_initial_words(self) -> None:
        """基于 self.token_counts 构建初始 words 与基础 vocab。
        - 字节 0-255 作为初始符号 ID
        - 特殊 token 分配额外 ID 并标记 frozen
        - 普通 token -> UTF-8 bytes -> list[int]
        """
        # 基础字节符号
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.bytes_to_id = {bytes([i]): i for i in range(256)}
        self.next_symbol_id = 256

        # 处理特殊 token
        special_id_map: Dict[str, int] = {}
        for tok in self.special_tokens:
            b = tok.encode('utf-8')
            if b in self.bytes_to_id:
                sid = self.bytes_to_id[b]
            else:
                sid = self.next_symbol_id
                self.next_symbol_id += 1
                self.id_to_bytes[sid] = b
                self.bytes_to_id[b] = sid
            special_id_map[tok] = sid

        self.words = []
        for token, cnt in self.token_counts.items():
            if cnt <= 0:
                continue
            if token in special_id_map:
                self.words.append(Word(symbols=[special_id_map[token]], count=cnt, frozen=True))
            else:
                b = token.encode('utf-8')
                self.words.append(Word(symbols=list(b), count=cnt, frozen=False))

    def _add_pair_occurrence(self, pair: tuple[int, int], wid: int, pos: int, weight: int) -> None:
        self.pair_freq[pair] = self.pair_freq.get(pair, 0) + weight
        bucket = self.pair_index.setdefault(pair, {})
        lst = bucket.setdefault(wid, [])
        lst.append(pos)
        # 推入堆（懒惰失效处理）
        freq = self.pair_freq[pair]
        if freq > 0:
            heapq.heappush(self.pair_heap, (-freq, pair[0], pair[1]))

    def _remove_pair_occurrence(self, pair: tuple[int, int], wid: int, pos: int, weight: int) -> None:
        """移除某个 pair 在指定词与位置上的出现（频次与索引）。
        如果 pair 或位置不存在则静默忽略（可能因重叠或之前已被处理）。"""
        bucket = self.pair_index.get(pair)
        if bucket is None:
            return
        pos_list = bucket.get(wid)
        if pos_list is None:
            return
        # 位置列表中删除该位置
        try:
            pos_list.remove(pos)
        except ValueError:
            # 不存在该位置，忽略
            pass
        if not pos_list:
            bucket.pop(wid, None)
        if not bucket:
            # 完全移除该 pair
            self.pair_index.pop(pair, None)
        # 频次减少
        if pair in self.pair_freq:
            self.pair_freq[pair] -= weight
            new_freq = self.pair_freq[pair]
            if new_freq <= 0:
                self.pair_freq.pop(pair, None)
            else:
                # 推入新快照
                heapq.heappush(self.pair_heap, (-new_freq, pair[0], pair[1]))

    def rebuild_pair_tables(self) -> None:
        """全量重建 pair_freq 与 pair_index (简单版本, 为保证正确性)."""
        self.pair_freq.clear()
        self.pair_index.clear()
        self.pair_heap.clear()
        for wid, w in enumerate(self.words):
            if w.frozen:
                continue
            syms = w.symbols
            if len(syms) < 2:
                continue
            for i in range(len(syms) - 1):
                pair = (syms[i], syms[i+1])
                self._add_pair_occurrence(pair, wid, i, w.count)

    # ------------------------ BPE 合并步骤 (简化实现) ------------------------ #
    def merge_one_step(self) -> bool:
        """执行一次最高频 pair 合并（增量更新 + 堆）。"""
        # 从堆中弹出直到找到有效 snapshot
        while self.pair_heap:
            negf, a, b = heapq.heappop(self.pair_heap)
            freq_current = self.pair_freq.get((a, b), 0)
            if freq_current > 0 and -negf == freq_current:
                best_pair = (a, b)
                best_freq = freq_current
                break
        else:
            return False
        a, b = best_pair  # type: ignore

        # 构造/获取新符号 ID
        new_bytes = self.id_to_bytes[a] + self.id_to_bytes[b]
        if new_bytes in self.bytes_to_id:
            new_id = self.bytes_to_id[new_bytes]
        else:
            new_id = self.next_symbol_id
            self.next_symbol_id += 1
            self.id_to_bytes[new_id] = new_bytes
            self.bytes_to_id[new_bytes] = new_id  # map bytes -> id
            self.symbol_meta[new_id] = (a, b)

        # 记录 merge（按字节对）
        self.merges.append((self.id_to_bytes[a], self.id_to_bytes[b]))

        # 受影响的词 ID 列表（复制）
        affected_word_ids = list(self.pair_index.get((a, b), {}).keys())

        # 删除 best_pair 的索引（稍后重新添加新生成的相关 pair）
        self.pair_index.pop((a, b), None)
        self.pair_freq.pop((a, b), None)

        for wid in affected_word_ids:
            w = self.words[wid]
            if w.frozen or len(w.symbols) < 2:
                continue
            weight = w.count
            old_syms = w.symbols
            L = len(old_syms)
            # 1) 移除该词所有旧 pair 贡献
            for i in range(L - 1):
                old_pair = (old_syms[i], old_syms[i+1])
                self._remove_pair_occurrence(old_pair, wid, i, weight)
            # 2) 构造新符号序列（合并 a,b）
            new_seq: List[int] = []
            i = 0
            while i < L:
                if i < L - 1 and old_syms[i] == a and old_syms[i+1] == b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(old_syms[i])
                    i += 1
            w.symbols = new_seq
            # 3) 添加新序列的 pair 贡献
            for i in range(len(new_seq) - 1):
                new_pair = (new_seq[i], new_seq[i+1])
                self._add_pair_occurrence(new_pair, wid, i, weight)

        return True

    def train_bpe(self) -> None:
        """根据目标 vocab_size 迭代执行 merge。"""
        # 目标是 vocab_size; 当前 vocab 为 id_to_bytes 大小
        while len(self.id_to_bytes) < self.vocab_size:
            progressed = self.merge_one_step()
            if not progressed:
                break

    # ------------------------ 导出接口 ------------------------ #
    def export_vocab(self) -> Dict[int, bytes]:
        return dict(self.id_to_bytes)

    def export_merges(self) -> List[tuple[bytes, bytes]]:
        return list(self.merges)

    # 调试：查看某个索引的符号序列（解码为 bytes 串）
    def debug_word_bytes(self, wid: int) -> bytes:
        parts = [self.id_to_bytes[sid] for sid in self.words[wid].symbols]
        return b''.join(parts)


def train_bpe_tokenizer(input_path: str | os.PathLike, vocab_size: int, special_tokens: List[str]) -> tuple[Dict[int, bytes], List[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer and return (vocab, merges).
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    # Deduplicate special tokens while preserving order
    dedup_specials = []
    seen = set()
    for tok in special_tokens:
        if tok not in seen:
            seen.add(tok)
            dedup_specials.append(tok)
    trainer = Tokenizer_trainer(input_path, vocab_size, dedup_specials)
    trainer.pretokenize_counts()
    print("pretokenization done")
    trainer.build_initial_words()
    # Initial pair tables
    trainer.rebuild_pair_tables()
    # Train (may early-stop if no merges possible)
    trainer.train_bpe()
    return trainer.export_vocab(), trainer.export_merges()


if __name__ == '__main__':
    input_path = 'data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000  
    special_tokens = ['<|endoftext|>']
    vocab, merges = train_bpe_tokenizer(input_path, vocab_size, special_tokens)
    with open("tokenizer/archive.pkl", "wb") as f:
        pkl.dump((vocab, merges), f)
    print(merges[-1000:])