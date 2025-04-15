import argparse
import numpy as np
import torch
import os
from einops import rearrange, einsum

from cs336_basics.model import transformer_lm, cross_entropy, softmax
from cs336_basics.optimizer import Adam, learning_rate_schedule, gradient_clipping
from cs336_basics.load_data import data_loading, save_checkpoint, load_checkpoint

def generate_text(model, 
    prompt_tokens, 
    max_new_tokens=50,
    temperature=1.0, 
    top_p=0.9, 
    end_of_text_token_id=None,
    device=None
):

    #model.eval()
    generated = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    generated = rearrange(generated, 'seq_len -> 1 seq_len')

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)
            # logits shape: (1, seq_len, vocab_size)
            next_logits = logits[:, -1, :]  

        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Convert to probabilities
        probs = softmax(next_logits, dim=-1).squeeze(0)
        # shape: (vocab_size,)

        # top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_idx = torch.searchsorted(cumulative_probs, top_p).item() + 1
        sorted_probs = sorted_probs[:cutoff_idx]
        sorted_indices = sorted_indices[:cutoff_idx]
        restricted_probs = sorted_probs / sorted_probs.sum()
        next_token = torch.multinomial(restricted_probs, num_samples=1)

        # map back to original token ID
        next_token_id = sorted_indices[next_token]

        generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

        if end_of_text_token_id is not None and next_token_id.item() == end_of_text_token_id:
            break

    return generated.squeeze(0).tolist()

def decode_tokens(token_ids, tokenizer):
    return tokenizer.decode(token_ids)
