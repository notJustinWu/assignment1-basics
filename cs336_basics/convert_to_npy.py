import numpy as np

def convert_txt_to_npy(txt_path, npy_path, delimiter=None):

    tokens = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.extend(map(int, line.split(delimiter)))

    arr = np.array(tokens, dtype=np.int64)
    np.save(npy_path, arr)
    print(f"Saved {len(tokens)} tokens to {npy_path}")

if __name__ == "__main__":
    convert_txt_to_npy("cs336_basics/vocab/TinyStoriesV2-GPT4-valid-encoded.txt", "cs336_basics/vocab/TinyStoriesV2-GPT4-valid-encoded.npy")