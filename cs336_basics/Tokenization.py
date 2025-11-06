from typing import List, Dict, Tuple, Optional, Iterable, Iterator
from .pretokenization_example import find_chunk_boundaries
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab: A dictionary mapping token IDs to byte sequences.
            merges: A list of tuples, where each tuple represents a merge operation (parent, child).
            special_tokens: An optional list of special token strings.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = None if special_tokens is None else special_tokens
        self.bytestoid = {
            v: k for k, v in vocab.items()
        }  # Create a reverse mapping from bytes to IDs

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and
        list of merges (in the same format that your BPE training code output) and (optionally)
        a list of special tokens.

        Args:
            vocab_filepath: Path to the vocabulary file.
            merges_filepath: Path to the merges file.
            special_tokens: An optional list of special token strings.

        Returns:
            An instance of the Tokenizer class.
        """

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: The input string to encode.

        Returns:
            A list of integer token IDs.
        """
        token_word = []

        # Step 1: Pre-tokenization (segment at special tokens)
        pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # 如果有special tokens，先split
        if self.special_tokens:
            # 创建split pattern
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = (
                "(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")"
            )
            # 在special tokens处分割文本
            text_segments = re.split(special_pattern, text)
        else:
            text_segments = [text]

        # 将分词的片段转化为int
        for segment in text_segments:
            if not segment:
                continue
            words = []
            # skip if it is a special_token
            if self.special_tokens and segment in self.special_tokens:
                special_token_bytes = segment.encode("utf-8")
                if special_token_bytes in self.bytestoid:
                    token_word.append(self.bytestoid[special_token_bytes])
                else:
                    print("No corresponding map for the special token")
                continue

            pre_tokens = pattern.findall(segment)

            # BPE Tokenization Process:
            # First, convert [the input] into a byte sequence.
            # Iterate through the merge rules according to the defined merge order, checking if that merge exists/is applicable.
            # Repeat until all merge rules have been attempted/checked.
            # Step 2 try merge bytes of the word in the order as when we construct the BPE algorithm
            for token in pre_tokens:
                word_bytes = token.encode(
                    "utf-8"
                )  # 这里先表示成为utf-8字符序列，便于判断如何合并
                word = [bytes([b]) for b in word_bytes]
                word = self._apply_merge(word)
                words.append(word)

            # Step 3 transform the list of bytes to list of token ids
            for word in words:
                for byte in word:
                    token_word.append(self.bytestoid[byte])

        return token_word

    def _apply_merge(self, word: str) -> List[bytes]:

        for merge in self.merges:
            if len(word) < 2:
                break

            i = 0
            while i < len(word) - 1:
                if word[i] == merge[0] and word[i + 1] == merge[1]:
                    merge_bytes = merge[0] + merge[1]
                    word = word[:i] + [merge_bytes] + word[i + 2 :]

                i += 1

        return word

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs. This is required for memory-efficient tokenization
        of large files that we cannot directly load into memory.

        Args:
            iterable: An iterable of strings.

        Returns:
            A generator that yields integer token IDs.
        """
        buffer = ""
        max_buffer_size = 8192  # 8KB 缓冲区

        for line in iterable:
            buffer += line

            # 当缓冲区足够大时，处理它
            if len(buffer) >= max_buffer_size:
                # 编码缓冲区内容
                ids = self.encode(buffer)
                for token_id in ids:
                    yield token_id
                buffer = ""  # 清空缓冲区

        # 处理剩余内容
        if buffer:
            ids = self.encode(buffer)
            for token_id in ids:
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded text string.
        """
        str_token: str = []
        byte_sequence = b""
        for id in ids:
            if self.vocab and id in self.vocab:
                byte_sequence += self.vocab[id]

        return byte_sequence.decode("utf-8", errors="replace")
