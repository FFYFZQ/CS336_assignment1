import os
from collections import Counter
import regex as re
from .pretokenization_example import find_chunk_boundaries


def merge_new_tokens(words, pair, new_token_id, pair_counts):
    """
    将所有words中的pair合并为new_token_id,并且更新原有的计数
    """
    new_words = []  # 创建一个新的用于返回的words，列表的每一项都是数字序列表示一个单词

    for word in words:
        if len(word) < 2:
            new_words.append(word)
            continue

        if pair[0] not in word or pair[1] not in word:
            new_words.append(word)
            continue

        new_word = []
        i = 0
        while i < len(word):
            if (
                i < len(word) - 1 and (word[i], word[i + 1]) == pair
            ):  # 如果匹配成功，直接加上new_token_id
                new_word.append(new_token_id)
                pair_counts[pair] -= 1

                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                if i < len(word) - 2:
                    pair_counts[(new_token_id, word[i + 2])] += 1
                    pair_counts[(word[i + 1], word[i + 2])] -= 1
                    if pair_counts[(word[i + 1], word[i + 2])] == 0:
                        del pair_counts[(word[i + 1], word[i + 2])]
                if i > 0:
                    pair_counts[(word[i - 1], new_token_id)] += 1
                    pair_counts[(word[i - 1], word[i])] -= 1
                    if pair_counts[(word[i - 1], word[i])] == 0:
                        del pair_counts[(word[i - 1], word[i])]
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_words.append(new_word)

    return new_words


def pre_tokenization(chunk: str, special_tokens, pair_counts, words):

    # 预分词的regex pattern
    pattern = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    # 如果有special tokens，先split
    if special_tokens:
        # 创建split pattern
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        # 在special tokens处分割文本
        text_segments = re.split(special_pattern, chunk)
    else:
        text_segments = [chunk]

    # 对每个分割后的片段分别进行预分词
    new_words = []
    for segment in text_segments:
        if not segment:  # 跳过空片段
            continue
        # 对这个片段进行预分词
        pre_tokens = pattern.findall(segment)
        for token in pre_tokens:
            word = list(token.encode("utf-8"))
            new_words.append(word)

    # 只对新添加的words进行计数
    for word in new_words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += 1

    # 将新words添加到全局words列表
    words.extend(new_words)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict, list]:
    """
    根据输入的文件内容训练BPE分词器，返回vocab和merge_bytes
    """

    # 建立token id到字节bytes的映射
    vocab = {i: bytes([i]) for i in range(0, 256)}
    # 先添加特殊token
    offset = 256
    for i, s_token in enumerate(special_tokens):
        vocab[i + offset] = s_token.encode("utf-8")

    offset = len(vocab)

    pair_counts = Counter()
    words = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 32, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pre_tokenization(chunk, special_tokens, pair_counts, words)

    # 确定merge的次数以及merge的tokens
    nums_merge = vocab_size - len(vocab)
    merges = []

    for j in range(nums_merge):

        best_pair = max(
            pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]])
        )

        if pair_counts[best_pair] <= 0:
            break

        merges.append(
            (vocab[best_pair[0]], vocab[best_pair[1]])
        )  # 添加获取的出现次数最多的元组

        new_token_bytes = (
            vocab[best_pair[0]] + vocab[best_pair[1]]
        )  # 一个新的字符序列(bytes)

        vocab[offset] = new_token_bytes  # 更新词汇表
        words = merge_new_tokens(
            words, best_pair, offset, pair_counts
        )  # merge的同时更新计数
        offset += 1

    return vocab, merges
