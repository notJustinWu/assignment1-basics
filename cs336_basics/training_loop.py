import argparse
import numpy as np
import torch
import os
from einops import rearrange, einsum
from tqdm import tqdm
from cs336_basics.model import transformer_lm, cross_entropy
from cs336_basics.optimizer import Adam, learning_rate_schedule, gradient_clipping
from cs336_basics.load_data import data_loading, save_checkpoint, load_checkpoint

def train_lm(train_tokens_path: str, 
    val_tokens_path: str,
    vocab_size: int, 
    context_length: int, 
    num_layers: int,
    d_model: int, 
    num_heads: int, 
    d_ff: int, 
    theta: float,
    max_steps: int, 
    batch_size: int,
    max_lr: float, 
    min_lr: float, 
    warmup_steps: int, 
    cosine_steps: int,
    weight_decay: float,   
    grad_clip: float, 
    device: str = "cuda",
    checkpoint_path: str = "checkpoint.pt", 
    validate_every: int = 500,
    save_every: int = 1000
):

    # load datasets using mmap but the dataset is in txt format
    # for now, train and valid are both in cs336_basics/vocab/TinyStoriesV2-GPT4-valid-encoded.txt
    train_tokens = np.load(train_tokens_path, mmap_mode='r')
    val_tokens = np.load(train_tokens_path, mmap_mode='r')

    # instantiate model
    model = transformer_lm(vocab_size=vocab_size,
        context_length=context_length, 
        num_layers=num_layers,
        d_model=d_model, 
        num_heads=num_heads, 
        d_ff=d_ff
    ).to(device)

    # instantiate optimizer
    optimizer = Adam(model.parameters(),
        lr=max_lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999), 
        eps=1e-8
    )

    # resume from checkpoint if it exists
    start_iteration = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading from checkpoint {checkpoint_path}...")
        start_iteration = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resumed training from iteration {start_iteration}.")

    # main training loop
    #model.train()
    for iteration in tqdm(range(start_iteration, max_steps)):
        # compute learning rate schedule
        lr_t = learning_rate_schedule(
            it=iteration,
            max_lr=max_lr,
            min_lr=min_lr,
            T_w=warmup_steps,
            T_c=cosine_steps
        )

        # update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_t

        # sample batch of data
        input_sequences, target_sequences = data_loading(
            x=train_tokens,
            batch_size=batch_size,
            context_length=context_length,
            device=device
        )

        # forward pass
        logits = model(input_sequences)

        # (d) Compute loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_2d = rearrange(logits, 'b s v -> (b s) v')
        targets_1d = rearrange(target_sequences, 'b s -> (b s)')
        loss = cross_entropy(logits_2d, targets_1d)

        # Backward + gradient clipping + optimizer step
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=grad_clip)
        optimizer.step()

        # logging loss
        if iteration % 50 == 0:
            print(f"Iteration: {iteration}, LR: {lr_t:.6g}, Loss: {loss.item():.4f}")

        if (iteration + 1) % validate_every == 0:
            val_loss = evaluate(model, val_tokens, context_length, device)
            print(f"Validation Iteration: {iteration}, Val Loss: {val_loss:.4f}")

        # save checkpoint
        if (iteration + 1) % save_every == 0:
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration + 1}")

    # final save
    save_checkpoint(model, optimizer, max_steps, checkpoint_path)
    print("Training complete. Final checkpoint saved.")

def evaluate(model, val_tokens, context_length, device, num_val_batches=10, batch_size=32):
    #model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_val_batches):
            inp, tgt = data_loading(val_tokens, batch_size, context_length, device)
            logits = model(inp)
            bsz, seq_len, vocab_sz = logits.shape
            logits_2d = rearrange(logits, 'b s v -> (b s) v')
            targets_1d = rearrange(tgt, 'b s -> (b s)')
            loss_val = torch.mean(cross_entropy(logits_2d, targets_1d))
            losses.append(loss_val.item())
    #model.train()
    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tokens", type=str,
                        default="cs336_basics/vocab/TinyStoriesV2-GPT4-valid-encoded.npy")
    parser.add_argument("--val_tokens", type=str,
                        default="cs336_basics/vocab/TinyStoriesV2-GPT4-valid-encoded.npy")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)

    # batch_size * max_steps * context_length = 32768000
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--cosine_steps", type=int, default=8000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--validate_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)

    args = parser.parse_args()

    train_lm(
        train_tokens_path=args.train_tokens,
        val_tokens_path=args.val_tokens,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        cosine_steps=args.cosine_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        validate_every=args.validate_every,
        save_every=args.save_every
    )

if __name__ == "__main__":
    main()
