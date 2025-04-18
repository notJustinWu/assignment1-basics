import argparse
import numpy as np
import torch
import os
from einops import rearrange, einsum

from cs336_basics.tokenizer_class import Tokenizer
from cs336_basics.model import transformer_lm, cross_entropy, softmax
from cs336_basics.optimizer import Adam, learning_rate_schedule, gradient_clipping
from cs336_basics.load_data import data_loading, save_checkpoint, load_checkpoint

def generate_text(model, 
    prompt_tokens, 
    max_new_tokens=50,
    temperature=1.0, 
    top_p=0.8, 
    end_of_text_token_id=None,
    device=None
):

    #model.eval()
    generated = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    generated = rearrange(generated, 'seq_len -> 1 seq_len')

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated, positions=torch.arange(generated.shape[1], device=device), theta=10000)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='/data/c-justwu/a1/checkpoints/owt/checkpoint.pt/checkpoint_40000.pt')
    parser.add_argument('--prompt', type=str, default = " ")
    parser.add_argument('--max_new_tokens', type=int, default = 256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    tokenizer = Tokenizer.from_files(
        cls=Tokenizer,
        vocab_filepath="/data/c-justwu/a1/owt_train_vocab.pkl",
        merges_filepath="/data/c-justwu/a1/owt_train_merges.pkl",
        special_tokens=["<|endoftext|>"]
    )

    model = transformer_lm(
        vocab_size=32000,
        context_length=256, 
        num_layers=4,
        d_model=512, 
        num_heads=16, 
        d_ff=1344
    ).to(device)

    optimizer = Adam(model.parameters(),
        lr=1e-3, 
        weight_decay=0.2,
        betas=(0.9, 0.999), 
        eps=1e-8
    )

    checkpoint = load_checkpoint(args.checkpoint_path, model, optimizer)
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    end_id = tokenizer.encode("<|endoftext|>")[0]

    generated_token_ids = generate_text(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, 
        top_p=args.top_p,
        end_of_text_token_id=end_id,
        device=device
    )

    generated_text = tokenizer.decode(generated_token_ids)
    print(generated_text)

if __name__ == "__main__":
    main()

