import torch
import numpy as np
import yaml
import argparse
import random
import os
from LM_basics.full_transformer import Transformer_LM
from loss_and_optimizer import cross_entropy, lr_cosine_schedule, gradient_clipping, AdamW
from loader_and_checkpoint import data_loader, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with config support")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--iterations", type=int, help="Training iterations")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--max_learning_rate", type=float, help="Max learning rate")
    parser.add_argument("--min_learning_rate", type=float, help="Min learning rate")

    parser.add_argument("--beta1", type=float, help="beta1 param of AdamW")
    parser.add_argument("--beta2", type=float, help="beta2 param of AdamW")
    parser.add_argument("--weight_decay", type=float, help="weight decay param of AdamW")

    parser.add_argument("--warmup_iters", type=int, help="Warmup iters")
    parser.add_argument("--cosine_cycle_iters", type=int, help="Cosine cycle iters")
    parser.add_argument("--gradient_clip_norm", type=float, help="Gradient Clip Norm")
    parser.add_argument("--save_interval", type=int, help="save_interval")
    parser.add_argument("--eval_interval", type=int, help="eval_interval")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoint_dir")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(args):
    # load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # command-line arguments cover in prior
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def evaluate(model, data, cfg, device, num_batches=10):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data_loader(data, cfg["batch_size"], cfg["context_length"], device)
            pred = model(x)
            loss = cross_entropy(pred, y)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    args = parse_args()
    cfg = load_config(args)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load training and validation data
    train_data = np.memmap("data/tokens_train.bin", dtype=np.uint16, mode="r")
    valid_data = np.memmap("data/tokens_valid.bin", dtype=np.uint16, mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model
    model = Transformer_LM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        rope_theta=cfg["RoPE_theta"]
    ).to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=cfg["max_learning_rate"],
                      betas=(cfg["beta1"], cfg["beta2"]),
                      weight_decay=cfg["weight_decay"])
    
    # Create checkpoint directory
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    
    # Resume from checkpoint if provided
    start_iteration = 0
    if args.resume:
        from loader_and_checkpoint import load_checkpoint
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    model.train()
    iteration = start_iteration
    best_val_loss = float('inf')
    
    print("Starting training...")
    # print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}") # 22,696,448
    
    try:
        while iteration < cfg.get("max_iterations", cfg["iterations"]):
            # Learning rate scheduling
            current_lr = lr_cosine_schedule(
                iteration, 
                cfg["max_learning_rate"], 
                cfg["min_learning_rate"], 
                cfg["warmup_iters"], 
                cfg["cosine_cycle_iters"]
            )
            
            # Update learning rate in optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Training step
            optimizer.zero_grad()
            x, y = data_loader(train_data, cfg["batch_size"], cfg["context_length"], device)
            pred = model(x)
            loss = cross_entropy(pred, y)
            loss.backward()
            
            # Gradient clipping
            gradient_clipping(model.parameters(), cfg["gradient_clip_norm"])
            
            optimizer.step()
            iteration += 1
            
            # Logging
            if iteration % 100 == 0:
                print(f"Iter {iteration}: loss={loss.item():.4f}, lr={current_lr:.2e}")
            
            # Evaluation
            if iteration % cfg["eval_interval"] == 0:
                val_loss = evaluate(model, valid_data, cfg, device)
                print(f"Iter {iteration}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
                
                # # Save best model
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     checkpoint_path = os.path.join(cfg["checkpoint_dir"], "best_model.pt")
                #     save_checkpoint(model, optimizer, iteration, checkpoint_path)
                #     print(f"Saved best model at iteration {iteration}")
            
            # Save regular checkpoints
            if iteration % cfg["save_interval"] == 0:
                checkpoint_path = os.path.join(cfg["checkpoint_dir"], f"checkpoint_{iteration}.pt")
                save_checkpoint(model, optimizer, iteration, checkpoint_path)
                print(f"Saved checkpoint at iteration {iteration}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    # Save final checkpoint
    final_checkpoint_path = os.path.join(cfg["checkpoint_dir"], "final_model.pt")
    save_checkpoint(model, optimizer, iteration, final_checkpoint_path)
    print(f"Training completed. Final model saved at {final_checkpoint_path}")


if __name__ == "__main__":
    main()