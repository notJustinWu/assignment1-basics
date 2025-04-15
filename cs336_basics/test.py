# read in the vocab and merges
import pickle

with open('cs336_basics/vocab/vocab_tiny_stories_gpt4_valid.pkl', 'rb') as f:
    vocab = pickle.load(f)
print(vocab)

