import pickle
import regex as re
from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator
import cProfile
from tqdm import tqdm
from multiprocessing import Pool
from typing import Dict, List, Tuple, Set
import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import time

# Parsing pattern used in GPT2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_bytes_temp = {}
        self.pretoken_to_id = {}
        for tok in self.special_tokens:
            tok_bytes = tok.encode('utf-8')
            self.special_bytes_temp[tok] = self.reverse_vocab[tok_bytes]

        # need to reverse special bytes so that we consider largest special tokens first
        self.special_bytes = dict(sorted(self.special_bytes_temp.items(), key=lambda item: len(item[0]), reverse=True))


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def segment_special_tokens(self, text: str) -> List[int]:
        i = 0
        result = []
        buffer = []
        n = len(text)

        while i < n:
            match = False
            for tok in self.special_bytes:
                if text.startswith(tok, i):
                    # first flush normal text
                    if buffer:
                        result.append((False, ''.join(buffer)))
                        buffer = []
                    # then add the special token
                    result.append((True, tok))
                    i += len(tok)
                    match = True
                    break

            if not match:
                buffer.append(text[i])
                i += 1
        if buffer:
            result.append((False, ''.join(buffer)))
        return result
    
    def apply_merges(self, word_bytes: List[bytes]) -> List[bytes]:
        for (a, b) in self.merges:
            i = 0
            merged_tokens = []
            while i < len(word_bytes):
                if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i+1]) == (a, b):
                    merged_tokens.append(a + b)
                    i += 2
                else:
                    merged_tokens.append(word_bytes[i])
                    i += 1  
            word_bytes = merged_tokens
        return word_bytes

    def encode(self, text: str) -> List[int]:
        # first segment the special tokens
        segments = self.segment_special_tokens(text)
        print(len(segments))
        # then encode the text
        encoded = []
        for is_special, token in segments: #tqdm(segments):
            if is_special:
                tok_id = self.special_bytes[token]
                encoded.append(tok_id)
            else:
                for word in re.finditer(PAT, token):
                    word = word.group(0)
                    if word in self.pretoken_to_id:
                        for b in self.pretoken_to_id[word]:
                            encoded.append(self.reverse_vocab[b])
                    else:
                        word_bytes = [bytes([b]) for b in word.encode('utf-8')]

                        # apply merges
                        word_bytes = self.apply_merges(word_bytes)
                        self.pretoken_to_id[word] = word_bytes

                        # encode the word
                        for b in word_bytes:
                            encoded.append(self.reverse_vocab[b])

        return encoded
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token
    
    def decode(self, ids: List[int]) -> str:
        decoded = []
        for id in ids:
            decoded.append(self.vocab[id])
        output_bytes = b''.join(decoded)
        return output_bytes.decode('utf-8', errors='replace')


# Parallel encoding
def encode_chunk(params):
    chunk_index, start, end, file_path, tokenizer = params
    with open(file_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8', errors='replace')
    encoded = tokenizer.encode(text)
    return (chunk_index, encoded)

def parallel_encode(file_path: str, tokenizer: Tokenizer, 
        special_token: str = "<|endoftext|>", num_chunks: int = None,
        sub_chunk_size: int = 5 * 1024 * 1024, processes: int = None) -> List[int]:
    
    if processes is None:
        processes = mp.cpu_count()

    if num_chunks is None:
        num_chunks = processes

    with open(file_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, processes, special_token.encode('utf-8'))
    
    tasks = []
    chunk_index = 0
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        current_pos = start
        while current_pos < end:
            chunk_end = min(current_pos + sub_chunk_size, end)
            tasks.append((chunk_index, current_pos, chunk_end, file_path, tokenizer))
            chunk_index += 1
            current_pos = chunk_end

    results = []
    with Pool(processes) as pool:
        for output in tqdm(pool.imap(encode_chunk, tasks), total=len(tasks)):
            results.append(output)

    results.sort(key=lambda x: x[0])
    all_ids = []
    for _, ids in results:
        all_ids.extend(ids)
    return all_ids


if __name__ == "__main__":
    #track time
    # print("Encoding tiny stories...")
    # start_time = time.time()
    # print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        cls=Tokenizer,
        vocab_filepath="cs336_basics/vocab/vocab_tiny_stories_gpt4_train.pkl",
        merges_filepath="cs336_basics/vocab/merges_tiny_stories_gpt4_train.pkl",
        special_tokens=["<|endoftext|>"]
    )
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # output_path = "cs336_basics/vocab/TinyStoriesV2-GPT4-train-encoded-iterable.txt"
    # print("parallel encoding")
    # start_time = time.time()
    # with open(input_path, "r", encoding="utf-8") as f:
    #     encoded = tokenizer.encode_iterable(f.read())
    # #encoded = parallel_encode(input_path, tokenizer, special_token="<|endoftext|>", num_chunks=10, sub_chunk_size=5_000_000, processes=8)

    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write(" ".join(map(str, encoded)))
    # i = 0
    # with open(input_path, "r", encoding="utf-8") as in_f, \
    #      open(output_path, "w", encoding="utf-8") as out_f:

    #     # get a generator of token IDs by streaming lines
    #     encoded_ids = tokenizer.encode_iterable(in_f)

    #     # write them out space-separated
    #     for tid in encoded_ids:
    #         out_f.write(str(tid))
    #         out_f.write(" ")
    #         i += 1
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    # print("Tiny stories encoded successfully.")
    # decode
    encoded_path = "cs336_basics/vocab/TinyStoriesV2-GPT4-train-encoded.txt"

    
    max_tokens_to_decode = 1000
    tokens_collected = 0
    encoded_tokens = []

    with open(encoded_path, "r", encoding="utf-8") as f:
        for line in f:
            token_strs = line.strip().split()

            for tstr in token_strs:
                encoded_tokens.append(int(tstr))
                tokens_collected += 1

                
                if tokens_collected >= max_tokens_to_decode:
                    break

            if tokens_collected >= max_tokens_to_decode:
                break


    decoded_text = tokenizer.decode(encoded_tokens)
    print(decoded_text)
