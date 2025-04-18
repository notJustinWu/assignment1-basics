import pickle
import regex as re
from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator
import cProfile
from tqdm import tqdm
from multiprocessing import Pool
from typing import Dict, List, Tuple, Set
import shutil
from numpy.lib.format import write_array_header_1_0
import os
from array import array
import numpy as np
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

        self.merges_dict = {}
        for rank, (a, b) in enumerate(self.merges):
            self.merges_dict.setdefault(a, {})[b] = rank
        
        self._PAT_RE = re.compile(PAT)

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
        while True:
            best_rank = None
            best_pos = None
            for i in range(len(word_bytes) - 1):
                a, b = word_bytes[i], word_bytes[i+1]
                rank = self.merges_dict.get(a, {}).get(b)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_pos is None:
                break
            word_bytes[best_pos:best_pos + 2] = [word_bytes[best_pos] + word_bytes[best_pos + 1]]
        return word_bytes

    def encode(self, text: str) -> List[int]:
        # first segment the special tokens
        segments = self.segment_special_tokens(text)
        #print(len(segments))
        # then encode the text
        encoded = []
        for is_special, token in segments: #tqdm(segments):
            if is_special:
                tok_id = self.special_bytes[token]
                encoded.append(tok_id)
            else:
                for word_match in self._PAT_RE.finditer(token):
                    word = word_match.group(0)
                    cached_ids = self.pretoken_to_id.get(word)
                    if cached_ids is not None:
                        encoded.extend(cached_ids)
                        continue
                    word_bytes = [bytes([b]) for b in word.encode('utf-8')]
                    word_bytes = self.apply_merges(word_bytes)
                    tok_ids = [self.reverse_vocab[b] for b in word_bytes]
                    self.pretoken_to_id[word] = tok_ids
                    encoded.extend(tok_ids)
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


import shutil
from numpy.lib.format import write_array_header_1_0

if __name__ == "__main__":
    print("Encoding owt train...")
    start_time = time.time()
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        cls=Tokenizer,
        vocab_filepath="cs336_basics/vocab/vocab_tiny_stories_gpt4_train.pkl",
        merges_filepath="cs336_basics/vocab/merges_tiny_stories_gpt4_train.pkl",
        special_tokens=["<|endoftext|>"]
    )

    input_path  = "data/TinyStoriesV2-GPT4-valid.txt"
    output_path = "cs336_basics/vocab/tinystories_valid_encoded.npy"

    tmp_path     = output_path + ".raw"  
    flush_every  = 100_000_000            
    token_buffer = array('H')             
    token_count  = 0

    print("Encoding & streaming …")
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(tmp_path,  "wb")            as fout:

        for tid in tokenizer.encode_iterable(fin):
            token_buffer.append(tid)
            token_count += 1

            if token_count % flush_every == 0:
                token_buffer.tofile(fout)     
                token_buffer = array('H')
                print(f"Pushed {token_count:,} tokens")

        if token_buffer:
            token_buffer.tofile(fout)

    print("Writing .npy header …")
    with open(tmp_path, "rb")  as raw_in , \
         open(output_path, "wb") as npy_out:

        header = {'descr': np.dtype(np.uint16).str, 'fortran_order': False, 'shape': (token_count,)}
        write_array_header_1_0(npy_out, header)
        shutil.copyfileobj(raw_in, npy_out)

    os.remove(tmp_path) 
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.1f} s")

# if __name__ == "__main__":
#     #track time
#     print("Encoding owt train...")
#     start_time = time.time()
#     print("Loading tokenizer...")
#     tokenizer = Tokenizer.from_files(
#         cls=Tokenizer,
#         vocab_filepath="cs336_basics/vocab/owt_train_vocab.pkl",
#         merges_filepath="cs336_basics/vocab/owt_train_merges.pkl",
#         special_tokens=["<|endoftext|>"]
#     )
#     input_path = "data/owt_train.txt"
#     output_path = "cs336_basics/vocab/owt_train_encoded1.npy"

#     max_tokens_to_decode = 1000


#     encoded_tokens = np.load('cs336_basics/vocab/owt_valid_encoded1.npy')

#     subset_tokens = encoded_tokens[:max_tokens_to_decode].tolist()

#     decoded_text = tokenizer.decode(subset_tokens)

#     print(decoded_text)


#     decoded_text = tokenizer.decode(encoded_tokens)
#     print(decoded_text)
