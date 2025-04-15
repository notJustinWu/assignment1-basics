import torch
import torch.nn as nn
import logging
import numpy as np
from einops import rearrange, einsum

def data_loading(x, batch_size, context_length, device):
    max_start = len(x) - context_length 
    start_indices = np.random.randint(0, max_start, batch_size)
    input_sequences = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    next_tokens = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    for i in range(batch_size):
        needed = x[start_indices[i]: start_indices[i] + context_length + 1]
        input_sequences[i] = torch.tensor(needed[:-1], dtype=torch.long)
        next_tokens[i] = torch.tensor(needed[1:], dtype=torch.long)
    return (input_sequences, next_tokens)

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    recover = torch.load(src)
    model.load_state_dict(recover['model_state'])
    optimizer.load_state_dict(recover['optimizer_state'])
    return recover['iteration']

