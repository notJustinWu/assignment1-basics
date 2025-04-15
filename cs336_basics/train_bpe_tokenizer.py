import json
import pickle
from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator
import cProfile
from typing import Dict, List, Tuple, Set
import os
from multiprocessing import Pool, cpu_count
import time
import pickle
import heapq
import regex as re
from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator
import cProfile
from multiprocessing import Pool
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import logging
import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(args):
    file_path, start, end, special_tokens, pattern = args
    counts = defaultdict(int)

    split_pattern = re.compile("|".join(re.escape(st) for st in special_tokens))
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start).decode('utf-8', errors='replace')
    
    chunks = re.split(split_pattern, chunk_data)
    for chunk in chunks:
        if chunk in special_tokens:
            continue
        for word in re.finditer(pattern, chunk):
            word = word.group(0)
            word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
            counts[word_bytes] += 1
    
    return counts

def parallel_pretokenize(input_path: str, special_tokens: List[str], num_processes: int) -> Dict[Tuple[bytes, ...], int]:
    print("Finding chunk boundaries...")
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token="<|endoftext|>".encode('utf-8'))

    CHUNK_SIZE = 1 * 1024 * 1024 

    tasks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        while start < end:
            next_end = min(start + CHUNK_SIZE, end)
            tasks.append((input_path, start, next_end, special_tokens, PAT))
            start = next_end

    print(f"Created {len(tasks)} total tasks.")
    print(f"Parallel pretokenizing using {num_processes} CPU workers...")

    final_counts = defaultdict(int)
    with Pool(num_processes) as pool:
        for partial_counts in tqdm(
            pool.imap_unordered(pretokenize_chunk, tasks), 
            total=len(tasks)
        ):
            for word, count in partial_counts.items():
                final_counts[word] += count

    return final_counts

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """
    Given the path to an input corpus, run train a BPE tokenizer and output its vocabulary and merges.

    Args:
        input_path: Path to BPE tokenizer training data.
        vocab_size: Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: A list of string special tokens to be added to the tokenizer vocabulary.

    Returns:
        Tuple of (vocab, merges):
            vocab: The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = {i: bytes([i]) for i in range(256)}

    # add special tokens 
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        vocab[len(vocab)] = token_bytes
    
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []
        
    token_counts = parallel_pretokenize(input_path, special_tokens, mp.cpu_count())
    #token_counts = pretokenize(input_path, special_tokens)
    
    # track merges
    merges = []
    pair_counts = defaultdict(int)
    for word, count in tqdm(token_counts.items()):
        for i in range(len(word) - 1):
            pair = word[i:i+2]
            pair_counts[pair] += count

    for _ in tqdm(range(num_merges)):
        # find the most frequent pair
        if not pair_counts:
            break
        
        most_frequent_pair, max_count = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0])
        )
        if max_count <= 0:
            break

        merges.append(most_frequent_pair)

        # add the most frequent pair to the vocab
        new_token_id = len(vocab)
        new_word = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[new_token_id] = new_word

        # merge this pair in all places that it shows up
        new_token_counts = defaultdict(int)
        for word, count in token_counts.items():
            i = 0
            new_word = word 
            while i < len(new_word) - 1:
                pair = new_word[i:i+2]
                if pair == most_frequent_pair:
                    prefix = new_word[:i]
                    suffix = new_word[i+2:]
                    merged = pair[0] + pair[1]
                    merged_tuple = (merged,)

                    # remove from pair_counts
                    pair_counts[most_frequent_pair] -= count

                    if i > 0:  
                        old_pair = (prefix[-1], pair[0])
                        new_pair = (prefix[-1], merged)
                        pair_counts[old_pair] -= count
                        pair_counts[new_pair] += count

                    if i + 2 < len(new_word):
                        old_pair = (pair[1], new_word[i+2])
                        new_pair = (merged, new_word[i+2])
                        pair_counts[old_pair] -= count
                        pair_counts[new_pair] += count

                    new_word = prefix + merged_tuple + suffix

                    i += 1  
                else:
                    i += 1
            
            new_token_counts[new_word] += count
        if pair_counts[most_frequent_pair] <= 0:
            del pair_counts[most_frequent_pair]
        token_counts = new_token_counts

    return vocab, merges

def train_tokenizer():
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    # serialize the vocab and merges
    with open("cs336_basics/vocab/vocab_tiny_stories_gpt4_valid.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("cs336_basics/vocab/merges_tiny_stories_gpt4_valid.pkl", "wb") as f:
        pickle.dump(merges, f)

if __name__ == "__main__":
    #track time
    start_time = time.time()
    print("Training tokenizer...")
    train_tokenizer()  
    end_time = time.time()
    print("Tokenizer trained successfully.")
    print(f"Time taken: {end_time - start_time} seconds")
