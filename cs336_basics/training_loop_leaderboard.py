import argparse
import numpy as np
import torch
import time
import os
from einops import rearrange, einsum
import wandb
from tqdm import tqdm
from cs336_basics.model import transformer_lm, cross_entropy, transformer_lm_ablation_no_rms_norm
from cs336_basics.model import transformer_lm_ablation_post_rms_norm, transformer_lm_ablation_silu
from cs336_basics.model import transformer_lm_weight_tying
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
    beta1: float,
    beta2: float,
    eps: float,
    warmup_steps: int,
    cosine_steps: int,
    weight_decay: float,
    grad_clip: float,
    wandb_project: str,
    device: str = "cuda",
    checkpoint_dir: str = "/data/c-justwu/a1/checkpoints/leaderboard_attempts/",
    validate_every: int = 500,
    save_every: int = 1000
):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    wandb.init(project=wandb_project)
    # load datasets using mmap but the dataset is in txt format
    # for now, train and valid are both in cs336_basics/vocab/TinyStoriesV2-GPT4-valid-encoded.txt
    train_tokens = np.load(train_tokens_path)#, mmap_mode='r')
    val_tokens = np.load(val_tokens_path)#, mmap_mode='r')

    # instantiate model
    model = transformer_lm_weight_tying(vocab_size=vocab_size,
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
        betas=(beta1, beta2), 
        eps=eps
    )
    #torch.compile(model)
    # resume from checkpoint if it exists
    start_iteration = 0

    # main training loop
    torch.cuda.empty_cache()
    model.train()
    start_time = time.time()
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
        batch_size, seq_len = input_sequences.shape
        positions = torch.arange(context_length, device=device)
        logits = model(input_sequences, positions=positions, theta=theta)

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

        wandb.log({
            "train_loss": loss.item(),
            "learning_rate": lr_t,
            "iteration": iteration,
            "wallclock_time": time.time() - start_time
        })

        # logging loss
        if iteration % 50 == 0:
            print(f"Iteration: {iteration}, LR: {lr_t:.6g}, Loss: {loss.item():.4f}")

        if (iteration + 1) % validate_every == 0:
            val_loss = evaluate(model, val_tokens, context_length, device, theta=theta)
            print(f"Validation Iteration: {iteration}, Val Loss: {val_loss:.4f}")
            wandb.log({
                "val_loss": val_loss,
                "iteration": iteration + 1,
                "wallclock_time": time.time() - start_time
            })
        # save checkpoint
        if (iteration + 1) % save_every == 0:
            save_checkpoint(model, optimizer, iteration + 1, os.path.join(checkpoint_dir, f"checkpoint_{iteration+1}.pt"))
            print(f"Checkpoint saved at iteration {iteration + 1}")

    # final save
    save_checkpoint(model, optimizer, max_steps, os.path.join(checkpoint_dir, f"checkpoint_{iteration+1}.pt"))
    print("Training complete. Final checkpoint saved.")

def evaluate(model, val_tokens, context_length, device, theta, num_val_batches=10, batch_size=64):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_val_batches):
            inp, tgt = data_loading(val_tokens, batch_size, context_length, device)
            positions = torch.arange(context_length, device=device)
            logits = model(inp, positions=positions, theta=theta)
            bsz, seq_len, vocab_sz = logits.shape
            logits_2d = rearrange(logits, 'b s v -> (b s) v')
            targets_1d = rearrange(tgt, 'b s -> (b s)')
            loss_val = torch.mean(cross_entropy(logits_2d, targets_1d))
            losses.append(loss_val.item())
    model.train()
    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tokens", type=str,
                        default="/data/c-justwu/a1/owt_train_encoded1.npy")
    parser.add_argument("--val_tokens", type=str,
                        default="/data/c-justwu/a1/owt_valid_encoded1.npy")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=500000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=int, default=1e-8)
    # batch_size * max_steps * context_length = 32768000
    parser.add_argument("--max_steps", type=int, default=40000)
    # 64 * 20000 * 256
    #try (1e-4, 1e-5), (3e-4, 3e-5), (1e-3, 1e-4), (3e-3, 3e-4), (1e-2, 1e-3), (1e-1, 1e-2)
    parser.add_argument("--max_lr", type=float, default=0.0015)
    parser.add_argument("--min_lr", type=float, default=0.00015)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--cosine_steps", type=int, default=18000)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/c-justwu/a1/checkpoints/leaderboard_attempts/checkpoint.pt")
    parser.add_argument("--wandb_project", type=str, default="owt")
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
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        wandb_project=args.wandb_project,
        warmup_steps=args.warmup_steps,
        cosine_steps=args.cosine_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        validate_every=args.validate_every,
        save_every=args.save_every
    )

if __name__ == "__main__":
    main()
