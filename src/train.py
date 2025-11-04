import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sentencepiece as spm
import random
import datetime 
from model import Transformer 
import matplotlib.pyplot as plt 

# 0. Random Seed Setup
def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. Learning Rate Scheduler (Noam Scheduler)
class NoamScheduler:
    """
    Implements the Noam scheduler: increases LR during warmup_steps, then decays.
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self._rate = 0
    
    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Calculate the current learning rate"""
        if step is None:
            step = self._step
        
        d_model_inv_sqrt = self.d_model ** (-0.5)
        step_inv_sqrt = step ** (-0.5) if step > 0 else 0
        warmup_inv_power = self.warmup_steps ** (-1.5)
        step_warmup_power = step * warmup_inv_power
        
        return d_model_inv_sqrt * min(step_inv_sqrt, step_warmup_power)

# 2. Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Training Program.")
    
    # Training Configuration
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--data_dir', type=str, default="./preprocessed", help='Directory for preprocessed data.')
    parser.add_argument('--save_dir', type=str, default="./results/checkpoints", help='Directory to save model checkpoints.')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Number of warmup steps for Noam scheduler.')
    parser.add_argument('--log_steps', type=int, default=200, help='Log training status every N steps.')
    
    # Validation data file name (dev_data.pt)
    parser.add_argument('--val_data_name', type=str, default="dev_data.pt", 
                        help='File name of the validation data (.pt file) inside data_dir.')

    # Random Seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.') 
    
    # Model Architecture Parameters
    parser.add_argument('--d_model', type=int, default=512, help='Embedding dimension.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers.')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward network dimension.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum sequence length.')
    parser.add_argument('--use_pos_enc', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Whether to use positional encoding.')
                        
    # Gradient Clipping (Stability trick)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping (set <= 0 to disable).')

    args = parser.parse_args()
    return args

# 3. Helper Classes and Functions

def count_parameters(model):
    """Counts the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TranslationDataset(Dataset):
    """PyTorch Dataset for loading preprocessed data"""
    def __init__(self, data):
        self.src = data["src"]
        self.tgt_in = data["tgt_in"]
        self.tgt_out = data["tgt_out"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt_in[idx], self.tgt_out[idx]

def get_data_loader(data_path, batch_size, shuffle=True):
    """Loads data and creates DataLoader"""
    data = torch.load(data_path)
    dataset = TranslationDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 4. Training Logic

def train_epoch(model, loader, criterion, scheduler, args, device, PAD_ID):
    """Executes one training epoch"""
    model.train()
    total_loss_sum = 0 # Cumulative loss for the entire epoch
    current_interval_loss_sum = 0 # Cumulative loss for the current log interval
    start_time = time.time()
    
    interval_losses = [] # List to store average loss for every log_steps
    steps_in_epoch = len(loader)
    
    for step, (src, tgt_in, tgt_out) in enumerate(tqdm(loader, desc="Training")):
        
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        scheduler.optimizer.zero_grad()
        
        output = model(src, tgt_in)
        
        # Calculate loss (output shape: [batch_size*seq_len, vocab_size], target shape: [batch_size*seq_len])
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)), 
            tgt_out.contiguous().view(-1)
        )
        
        loss.backward()
        
        # Gradient Clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) 
        
        scheduler.step()

        step_loss = loss.item()
        total_loss_sum += step_loss
        current_interval_loss_sum += step_loss
        
        if (step + 1) % args.log_steps == 0:
            elapsed_time = time.time() - start_time
            
            interval_avg_loss = current_interval_loss_sum / args.log_steps
            interval_losses.append(interval_avg_loss)
            
            current_lr = scheduler._rate
            
            print(f"| Step {step+1}/{steps_in_epoch} | Loss {interval_avg_loss:.4f} | LR {current_lr:.6e} | Time {elapsed_time:.2f}s")
            
            current_interval_loss_sum = 0
            start_time = time.time()
            
    epoch_avg_loss = total_loss_sum / steps_in_epoch
    
    # Handle the last, possibly incomplete log interval
    remaining_steps = steps_in_epoch % args.log_steps
    if remaining_steps > 0 and current_interval_loss_sum > 0:
        final_interval_avg_loss = current_interval_loss_sum / remaining_steps
        interval_losses.append(final_interval_avg_loss)
    
    return epoch_avg_loss, interval_losses

# 5. Validation Logic

@torch.no_grad() # Ensure no gradient calculation
def validate_epoch(model, loader, criterion, device, PAD_ID):
    """Executes a validation epoch, calculates average validation loss"""
    model.eval() # Set to evaluation mode
    total_loss_sum = 0
    
    for src, tgt_in, tgt_out in tqdm(loader, desc="Validating"):
        # 1. Move data to device
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        
        # 2. Forward pass
        output = model(src, tgt_in)
        
        # 3. Calculate Loss
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)), 
            tgt_out.contiguous().view(-1)
        )
        total_loss_sum += loss.item()
            
    # Calculate the average loss for the entire epoch
    if len(loader) == 0:
        return 0.0 # Avoid division by zero
        
    epoch_avg_loss = total_loss_sum / len(loader)
    return epoch_avg_loss

# 6. Plotting Function
def plot_losses(history, filename):
    """Plots training and validation loss curves and saves them as a file"""
    plt.figure(figsize=(10, 6))
    
    # Plot Training Loss
    if history['train_losses']:
        plt.plot(history['steps'], history['train_losses'], 
                 label='Training Loss (Interval Avg)', 
                 color='blue', 
                 alpha=0.7)

    # Plot Validation Loss
    if history.get('valid_losses') and history.get('valid_steps'):
        # Validation loss only has one point per epoch, marked with circles and dashed lines
        plt.plot(history['valid_steps'], history['valid_losses'], 
                 label='Validation Loss (End of Epoch)', 
                 color='red', 
                 marker='o', # Use circle markers
                 linestyle='--') 
        
    # Use Run ID for tracking
    run_id_display = history.get('run_id', 'UnknownID')
    
    title_str = (
        f'Loss Curve (L{history["L"]}/D{history["D"]}/H{history["H"]}/'
        f'{("pos" if history["use_pos_enc"] else "nopos")}/Seed{history["seed"]}/Run{run_id_display})'
    )
    
    # Adjust title to show current step progress
    max_step = history['steps'][-1] if history['steps'] else 0
    title_str += f' - Up to Step {max_step}'

    plt.title(title_str)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Ensure the save path exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"📈 Loss plot saved to {filename}")

# 7. Main Function

def main():
    args = parse_args()
    
    # Generate a unique run ID (timestamp) to prevent file overwrites
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"✅ Current Run ID (Timestamp): {run_timestamp}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")

    # 1. Data Loading
    try:
        train_data_path = os.path.join(args.data_dir, "train_data.pt")
        vocab_info_path = os.path.join(args.data_dir, "vocab_info.json")

        if not os.path.exists(train_data_path) or not os.path.exists(vocab_info_path):
            print(f"❌ Cannot find preprocessed data in {args.data_dir}. Please run data.py first.")
            return

        # Load Training Data
        train_loader = get_data_loader(train_data_path, args.batch_size, shuffle=True)
        
        # 1.1 Load Validation Data (Crucial for simultaneous output)
        val_loader = None
        val_data_path = os.path.join(args.data_dir, args.val_data_name)
        if os.path.exists(val_data_path):
            # Do not shuffle validation set
            val_loader = get_data_loader(val_data_path, args.batch_size, shuffle=False)
            print(f"✅ Validation data loaded from {val_data_path}.")
        else:
            print(f"⚠️ Warning: Validation data not found at {val_data_path}. Skipping validation.")

        with open(vocab_info_path, "r") as f:
            vocab_info = json.load(f)
        
        sp_path = os.path.join(args.data_dir, "spm_bpe_8k.model")
        if not os.path.exists(sp_path):
            print(f"❌ Cannot find SentencePiece model at {sp_path}.")
            return
        
        sp = spm.SentencePieceProcessor(model_file=sp_path)
        
        VOCAB_SIZE = vocab_info["vocab_size"]
        PAD_ID = vocab_info.get("pad_id", sp.pad_id())
        
        print(f"✅ Data loaded. Vocab Size: {VOCAB_SIZE}, PAD ID: {PAD_ID}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Model Initialization
    model = Transformer(
        src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE, 
        d_model=args.d_model, num_heads=args.num_heads, 
        num_layers=args.num_layers, d_ff=args.d_ff, 
        dropout=args.dropout, max_len=args.max_len,
        use_pos_enc=args.use_pos_enc 
    ).to(DEVICE)
    
    print("🚀 Model initialized.")
    n_params = count_parameters(model)
    print(f"📊 Total trainable parameters: {n_params:,}")

    # 3. Optimizer, Loss Function, and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    scheduler = NoamScheduler(
        optimizer, 
        d_model=args.d_model, 
        warmup_steps=args.warmup_steps
    )

    # CrossEntropy Loss, ignoring PAD_ID
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    # 4. Training Loop Setup
    best_loss = float('inf')
    
    # Base filename now includes the unique run timestamp
    pos_enc_str = 'pos' if args.use_pos_enc else 'nopos'
    base_filename = (
        f"transformer_L{args.num_layers}_D{args.d_model}_H{args.num_heads}_{pos_enc_str}_seed{args.seed}_{run_timestamp}"
    )
    
    # Training history tracking
    history = {
        'steps': [], 'train_losses': [], 
        'valid_steps': [], 'valid_losses': [],  
        'L': args.num_layers, 'D': args.d_model, 'H': args.num_heads,
        'seed': args.seed, 'use_pos_enc': args.use_pos_enc, 
        'log_steps': args.log_steps,
        'total_steps': 0,
        'run_id': run_timestamp 
    }
    
    steps_per_epoch = len(train_loader)

    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # 5. Training Phase
        epoch_avg_loss, interval_losses = train_epoch(model, train_loader, criterion, scheduler, args, DEVICE, PAD_ID)
        
        # Record training loss history
        start_step_base = history['total_steps']
        current_step_counter = start_step_base
        for i, loss_val in enumerate(interval_losses):
            if i < len(interval_losses) - 1:
                current_step_counter += args.log_steps
            else:
                current_step_counter = start_step_base + steps_per_epoch
            
            history['steps'].append(current_step_counter)
            history['train_losses'].append(loss_val)

        # Update total steps
        history['total_steps'] += steps_per_epoch

        # 6. Validation Phase (Simultaneous Output)
        valid_avg_loss = None
        if val_loader is not None:
            # ONLY validate if the validation data was successfully loaded
            valid_avg_loss = validate_epoch(model, val_loader, criterion, DEVICE, PAD_ID)
            
            # Record validation loss
            history['valid_steps'].append(history['total_steps'])
            history['valid_losses'].append(valid_avg_loss)
            
            # CRUCIAL: Simultaneous Loss Output
            print(f"✅ End of Epoch {epoch} | Train Loss (Avg): {epoch_avg_loss:.4f} | VALIDATION Loss (Avg): {valid_avg_loss:.4f}")
            
        else:
            print(f"End of Epoch {epoch} | Train Loss (Avg): {epoch_avg_loss:.4f}")

        # 7. Model Checkpoint (based on Validation Loss if available)
        current_loss_to_check = valid_avg_loss if valid_avg_loss is not None else epoch_avg_loss
        
        if current_loss_to_check < best_loss:
            best_loss = current_loss_to_check
            
            checkpoint_name = f"{base_filename}_epoch{epoch:03d}_checkloss{best_loss:.4f}.pt"
            checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
            
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved BEST model to {checkpoint_path} (Loss: {best_loss:.4f})")
            
        # 8. Periodic Plotting (Every 10 epochs or the last epoch)
        if epoch % 10 == 0 or epoch == args.epochs:
            plot_filename = os.path.join(results_dir, f"{base_filename}_loss_curve_e{epoch:03d}.png")
            plot_losses(history, plot_filename)


if __name__ == '__main__':
    main()
