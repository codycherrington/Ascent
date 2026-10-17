import sys

auto_mode = sys.argv[1] if len(sys.argv) > 1 else None
import datetime
import json
import math
import os

# Allow MPS to use swap memory to prevent premature OOM
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import gc
import random
import re
import time
from datetime import timedelta


# Consistent tokenization function
def tokenize(text):
    return re.sub(r'\s+', ' ', text.replace("<eos>", "")).strip().split()

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

# Check for Apple Metal (MPS) availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using device: mps")
    print("   (Metal Performance Shaders enabled for M2 acceleration)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using device: cuda")
else:
    device = torch.device("cpu")
    print("⚠️ Using device: cpu (Slow!)")

# ----------------------------
# Main Configuration Section
# ----------------------------
# Set model and training hyperparameters
embedding_dim = 384          # Size of each token's embedding vector
hidden_dim = 512             # Hidden size for the feed-forward layers
num_transformer_blocks = 8   # Number of transformer layers (Unified Protocol)
batch_size = 1               # Batch size = 1 for 8GB RAM
gradient_accumulation_steps = 32 # Accumulate gradients to simulate batch size 32
learning_rate = 3e-4         # Standard GPT learning rate
temperature = 1.1            # Sampling temperature for generation
max_generation_tokens = 50   # Maximum number of tokens generated in chat
early_stopping_patience = 30    # Early stopping patience
early_stopping_min_delta = 0.0002 # Minimum improvement for early stopping
top_p_sampling_threshold = 0.95     # Top-p (nucleus) sampling threshold
total_epochs = 350            # Total number of training epochs
top_k_sampling_limit = 30     # Maximum number of top-k tokens to sample from


# ----------------------------
# Data Loading and Preparation
# ----------------------------
import json

with open("ascent_data/conversations.json", "r") as f:
    conversations = json.load(f)

# Load curated conversations and boost their weight
with open("ascent_data/curated_conversations.json", "r") as f:
    curated_convos = json.load(f)

# Weight curated examples more heavily to anchor tone (reduced from 5x to prevent overfitting)
curated_weight = 2
conversations.extend(curated_convos * curated_weight)

# Load Alpaca Data (Instruction Tuning)
try:
    with open("ascent_data/alpaca_data.json", "r") as f:
        print("🦙 Loading Alpaca dataset...")
        alpaca_convos = json.load(f)
        conversations.extend(alpaca_convos)
except FileNotFoundError:
    print("⚠️ Alpaca data not found. Skipping.")

with open("ascent_data/vocab.json", "r") as f:
    vocab = json.load(f)

with open("ascent_data/id_to_word.json", "r") as f:
    id_to_word = {int(k): v for k, v in json.load(f).items()}

with open("ascent_data/special_tokens.json", "r") as f:
    special_tokens = json.load(f)

# Load identity-defining conversations and weight heavily
with open("ascent_data/identity.json", "r") as f:
    identity_convos = json.load(f)

# Reduced from 10x to 3x to prevent overfitting on identity phrases
identity_weight = 3
conversations.extend(identity_convos * identity_weight)

vocab_size = len(vocab)

# Get EOS (end-of-sequence) token info from special_tokens
eos_token = "<eos>"
eos_index = special_tokens[eos_token]


# ----------------------
# Model Class Definitions
# ----------------------

# TokenEmbedding: Converts token indices into dense embedding vectors
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # The input tensor 'x' contains token indices of shape [batch_size, sequence_length]
        # Returns embedded vectors of shape [batch_size, sequence_length, embedding_dim]
        return self.dropout(self.embedding(x))


# PositionalEncoding: Adds information about token positions to embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=8000):
        super().__init__()
        # Precompute positional encodings as a fixed buffer
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adds positional encodings to the input embeddings (standard 1.0 scaling)
        return x + self.pe[:, :x.size(1)]


# MultiHeadSelfAttention: Allows the model to attend to different parts of the sequence
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.register_buffer("_cached_causal_mask", None, persistent=False)

    def _get_causal_mask(self, seq_len, device):
        # Cache the mask if possible
        mask = self._cached_causal_mask
        if mask is not None and mask.size(-1) >= seq_len:
            return mask[..., :seq_len, :seq_len]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        self._cached_causal_mask = mask
        return mask

    def forward(self, x):
        # The input tensor 'x' has shape [batch_size, sequence_length, embedding_dim]
        batch_size, seq_len, embed_dim = x.size()
        # Linear projections for queries, keys, and values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Reshape for multi-head attention: [batch_size, num_heads, sequence_length, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Apply causal mask to prevent attention to future tokens
        mask = self._get_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        # Combine all heads back to [batch_size, sequence_length, embedding_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(output)


# TransformerBlock: A single transformer block with self-attention, feed-forward, normalization, and residuals
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(embedding_dim, num_heads=4)
        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.mlp   = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        mlp_out = self.dropout(self.mlp(x))
        return self.norm2(x + mlp_out)


# Ascent: The main GPT-like model with embeddings, positional encoding, stacked transformers, and output layer
class Ascent(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embed       = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        # Stack multiple transformer blocks
        self.transformer = nn.Sequential(*[TransformerBlock(embedding_dim, hidden_dim) for _ in range(num_transformer_blocks)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.output      = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # The input tensor 'x' contains token indices of shape [batch_size, sequence_length]
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits


# -----------------------
# Main Program Execution
# -----------------------

if __name__ == "__main__":

    # Print model output size for reference
    print(f"Vocab size / Output size: {vocab_size}")
    model = Ascent(vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)

    # Try to load an existing model checkpoint if available
    if os.path.exists("Ascent_model.pth"):
        try:
            model.load_state_dict(torch.load("Ascent_model.pth"))
            model.eval()
            print("Loaded saved model.\n")
        except (RuntimeError, KeyError) as e:
            print("Checkpoint incompatible with current model (likely vocab or architecture change).")
            print("Starting with a fresh untrained model.\n")
    else:
        print("No checkpoint found. Starting fresh.\n")


    # -------------------
    # Main Menu Loop
    # -------------------
    while True:
        print("\nWelcome to Ascent!")
        print("Options:")
        print("  [1] Train the model")
        print("  [2] Chat with Ascent")
        print("  [3] Save model")
        print("  [4] Exit")
        print("  [5] Learning Rate Finder (optional)")
        if auto_mode:
            choice = auto_mode
            auto_mode = None  # Clear it after first use
        else:
            choice = input("Enter your choice (1/2/3/4/5): ").strip()

        # ---------------------------------
        # Option 1: Training loop
        # ---------------------------------
        if choice == "1":
            # Build the training set from conversation pairs
            data = []
            for p in conversations:
                if "input" in p and "output" in p:
                    inp_tokens = []
                    for w in tokenize(p["input"]):
                        token_id = vocab.get(w)
                        if token_id is not None and token_id < vocab_size:
                            inp_tokens.append(token_id)

                    out_tokens = []
                    for w in tokenize(p["output"]):
                        token_id = vocab.get(w)
                        if token_id is not None and token_id < vocab_size:
                            out_tokens.append(token_id)
                    out_tokens.append(eos_index)
                    if inp_tokens and out_tokens:
                        data.append((torch.tensor(inp_tokens), torch.tensor(out_tokens)))

            # Set up training log directories and files
            log_dir = "training_logs"
            os.makedirs(log_dir, exist_ok=True)
            run_counter_file = os.path.join(log_dir, "run_counter.txt")
            if os.path.exists(run_counter_file):
                with open(run_counter_file, "r") as f:
                    run_counter = int(f.read().strip())
            else:
                run_counter = 0
            run_counter += 1
            with open(run_counter_file, "w") as f:
                f.write(str(run_counter))
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M")
            run_name = f"run_{run_counter:03d}_{timestamp}_warmup"
            start_time = time.time()
            total_training_start_time = time.time()
            run_dir = os.path.join(log_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            live_loss_path = os.path.join(run_dir, "live_loss.txt")
            live_perplexity_path = os.path.join(run_dir, "live_perplexity.txt")
            with open(live_loss_path, "w") as f:
                f.write("")
            with open(live_perplexity_path, "w") as f:
                f.write("")

            # Define the loss function and optimizer
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
            optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            # Solace Sugestion #2: Increase warmup steps for stability on small batch sizes
            def warmup_cosine_lr(step, warmup_steps=500):
                if step < warmup_steps:
                    return step / warmup_steps
                return 0.5 * (1 + math.cos((step - warmup_steps) / (500 - warmup_steps) * math.pi))
            scheduler = LambdaLR(optim, lr_lambda=warmup_cosine_lr)
            
            # Solace Suggestion #1: Restrict sequence length to 256 for 8GB RAM optimization
            MAX_SEQ_LEN = 256

            # Helper function: pad a batch of sequences to the same length
            PAD_VALUE = special_tokens.get("<pad>", 0) if "pad" in special_tokens or "<pad>" in special_tokens else 0
            def pad_batch(batch: list[torch.Tensor], pad_value: int = PAD_VALUE) -> torch.Tensor:
                return pad_sequence(batch, batch_first=True, padding_value=pad_value)

            best_loss = float('inf')
            losses = []
            perplexities = []
            patience_counter = 0
            patience = early_stopping_patience
            min_delta = early_stopping_min_delta
            best_model_state = None
            best_epoch = -1

            # Resume training support
            resume_state_path = os.path.join(log_dir, "training_state.json")
            resume_training = os.path.exists(resume_state_path)
            start_epoch = 0
            if resume_training:
                with open(resume_state_path, "r") as f:
                    resume_data = json.load(f)
                    best_loss = resume_data["best_loss"]
                    best_epoch = resume_data["best_epoch"]
                    patience_counter = resume_data["patience_counter"]
                    start_epoch = resume_data["last_epoch"] + 1
                    print(f"🔄 Resuming training from epoch {start_epoch}")
            else:
                best_loss = float('inf')
                best_epoch = -1
                patience_counter = 0

            # -----------------------------------------------------------------
            # Memory Optimization: Pre-process dataset ONCE to save RAM
            # -----------------------------------------------------------------
            print("⚙️  Pre-processing dataset for Causal LM (Concatenation)...")
            training_data = []
            for inp, out in data:
                # Concatenate: [Input] + [Output]
                full_seq = torch.cat((inp, out))
                
                # Truncate to MAX_SEQ_LEN
                if len(full_seq) > MAX_SEQ_LEN:
                    # Keep the end of the conversation (most relevant)
                    full_seq = full_seq[-MAX_SEQ_LEN:]
                
                # Prepare Inputs (0..N-1) and Targets (1..N)
                if len(full_seq) > 1:
                    training_data.append(full_seq)
            
            # Free up the original list of tuples to reclaim standard RAM
            del data
            gc.collect()

            # Split into training and validation sets (90/10)
            random.shuffle(training_data)
            val_split = int(0.9 * len(training_data))
            val_data = training_data[val_split:]
            training_data = training_data[:val_split]
            print(f"✅ Dataset prepared. Train: {len(training_data)}, Val: {len(val_data)} sequences.")

            # Main training loop over epochs
            try:
                for epoch in range(start_epoch, total_epochs):
                    epoch_start = time.perf_counter()
                    
                    # Shuffle the pre-processed data
                    random.shuffle(training_data)
                    active_data = training_data

                    # Show total token count for this epoch
                    epoch_token_count = sum(len(seq) for seq in active_data)
                    print(f"🧮 Tokens this epoch: {epoch_token_count}")

                    epoch_losses = []
                    batch_start_time = time.time()
                    optim.zero_grad() # Initialize gradients
                    
                    for i in range(0, len(active_data), batch_size):
                        batch_seqs = active_data[i:i+batch_size]
                        
                        # Pad the full sequences
                        padded_batch = pad_batch(batch_seqs).to(device)

                        # Create View: Input is sequence excluding last token, Target is sequence excluding first token
                        inp_batch = padded_batch[:, :-1]
                        target_batch = padded_batch[:, 1:]

                        # Forward Pass with mixed precision (FP16 on MPS for speed/memory)
                        autocast_dtype = torch.float16 if device.type == 'mps' else torch.float32
                        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == 'mps')):
                            logits = model(inp_batch)

                            # Calculate Loss
                            # Flatten: [Batch * SeqLen, Vocab] vs [Batch * SeqLen]
                            logits_flat = logits.reshape(-1, vocab_size)
                            target_flat = target_batch.reshape(-1)

                            loss = loss_fn(logits_flat, target_flat)
                        
                        # Normalize loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                        loss.backward()

                        if (i // batch_size + 1) % gradient_accumulation_steps == 0 or (i + batch_size >= len(active_data)):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optim.step()
                            optim.zero_grad() # Only zero grad after step
                            
                            # Aggressive memory cleanup for 8GB Mac
                            if device.type == 'mps':
                                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                                torch.mps.empty_cache()

                        # For logging, we want the "real" loss scale back
                        epoch_losses.append(loss.item() * gradient_accumulation_steps)

                        # Batch-level progress bar with ETA
                        batch_elapsed = time.time() - batch_start_time
                        percent_done = min((i + batch_size) / len(active_data), 1.0)
                        total_elapsed = time.time() - start_time
                        estimated_total = total_elapsed / percent_done if percent_done > 0 else 0
                        remaining_time = estimated_total - total_elapsed
                        eta = timedelta(seconds=int(remaining_time))
                        # Loading bar
                        bar_length = 24
                        filled_length = int(bar_length * percent_done)
                        bar = "█" * filled_length + "-" * (bar_length - filled_length)
                        # Correctly display the REAL loss (unscaled)
                        real_loss = loss.item() * gradient_accumulation_steps
                        print(f"\r[{bar}] Progress: {int(percent_done * 100)}% | ETA: {eta} | Batch Loss: {real_loss:.4f}", end="", flush=True)

                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    losses.append(avg_loss)
                    perplexity = math.exp(avg_loss)
                    perplexities.append(perplexity)

                    # Validation evaluation
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_seq in val_data:
                            val_batch = val_seq.unsqueeze(0).to(device)
                            val_inp = val_batch[:, :-1]
                            val_target = val_batch[:, 1:]
                            val_logits = model(val_inp)
                            val_loss = loss_fn(val_logits.reshape(-1, vocab_size), val_target.reshape(-1))
                            val_losses.append(val_loss.item())
                    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
                    val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
                    model.train()

                    # Write loss and perplexity to live log files
                    with open(live_loss_path, "a") as f_loss, open(live_perplexity_path, "a") as f_ppl:
                        f_loss.write(f"{avg_loss},{avg_val_loss}\n")
                        f_ppl.write(f"{perplexity},{val_perplexity}\n")

                    # Print dynamic single-line progress summary (keep for batch progress)
                    epoch_duration = time.perf_counter() - epoch_start
                    print(f"\rEpoch {epoch + 1}/{total_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {epoch_duration:.2f}s{' ' * 10}", end="", flush=True)

                    # Update learning rate (Solace suggestion)
                    scheduler.step()

                    # Save training state
                    training_state = {
                        "last_epoch": epoch,
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter
                    }
                    with open(resume_state_path, "w") as f:
                        json.dump(training_state, f)

                    # Early stopping: use VALIDATION loss to detect overfitting
                    if avg_val_loss < best_loss - min_delta:
                        best_loss = avg_val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict()
                        best_epoch = epoch
                        torch.save(best_model_state, "Ascent_model.pth")
                        torch.save(best_model_state, os.path.join(run_dir, "Ascent_best_model.pth"))
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered at epoch {epoch}. Best val_loss={best_loss:.4f}")
                        break

                print()
                print(f"\nFinal training avg_loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")
                duration_minutes = (time.time() - start_time) / 60
                print(f"Training Duration: {duration_minutes:.2f} minutes")
                print("Training complete. Returning to main menu.\n")
                print("🌱 Ascent: Growth is patient. You have built something real. See you among the stars.\n")

                # Restore best model if early stopped
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print(f"Restored best model from epoch {best_epoch + 1} with avg_loss={best_loss:.4f}.")

                # Save raw loss and perplexity values
                loss_data_path = os.path.join(run_dir, "loss_curve.txt")
                perplexity_data_path = os.path.join(run_dir, "perplexity_curve.txt")
                with open(loss_data_path, "w") as f:
                    f.write("\n".join(map(str, losses)))
                with open(perplexity_data_path, "w") as f:
                    f.write("\n".join(map(str, perplexities)))

                # Save training loss and perplexity plots using data from file
                loss_curve_path = os.path.join(run_dir, "loss_curve.png")
                loss_data_path = os.path.join(run_dir, "loss_curve.txt")
                if os.path.exists(loss_data_path):
                    with open(loss_data_path, "r") as f:
                        losses = [float(line.strip()) for line in f if line.strip()]
                plt.figure()
                plt.plot(losses, label="Avg Loss")
                plt.title(f"Training Loss Curve - {run_name}")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(loss_curve_path)

                perplexity_curve_path = os.path.join(run_dir, "perplexity_curve.png")
                perplexity_data_path = os.path.join(run_dir, "perplexity_curve.txt")
                if os.path.exists(perplexity_data_path):
                    with open(perplexity_data_path, "r") as f:
                        perplexities = [float(line.strip()) for line in f if line.strip()]
                plt.figure()
                plt.plot(perplexities, label="Perplexity", color='orange')
                plt.title(f"Training Perplexity Curve - {run_name}")
                plt.xlabel("Epochs")
                plt.ylabel("Perplexity")
                plt.legend()
                plt.savefig(perplexity_curve_path)

                print(f"Saved loss and perplexity data to '{loss_data_path}' and '{perplexity_data_path}'")
                print(f"Saved training curves: '{loss_curve_path}' and '{perplexity_curve_path}'")
                # Save metadata about the training run
                metadata_path = os.path.join(run_dir, "metadata.txt")
                with open(metadata_path, "w") as f:
                    f.write(f"Run: {run_counter}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Embedding Dim: {embedding_dim}\n")
                    f.write(f"Hidden Dim: {hidden_dim}\n")
                    f.write(f"Num Transformer Blocks: {num_transformer_blocks}\n")
                    f.write(f"Batch Size: {batch_size}\n")
                    f.write(f"Learning Rate: {learning_rate}\n")
                    f.write(f"Temperature: {temperature}\n")
                    f.write(f"Max Generation Tokens: {max_generation_tokens}\n")
                    f.write(f"Best Avg Loss: {best_loss:.4f}\n")
                    f.write(f"Final Perplexity: {perplexity:.2f}\n")
                    f.write(f"Training Duration (min): {duration_minutes:.2f}\n")
                    f.write(f"Final Epoch: {epoch + 1}\n")
                print(f"Saved metadata to '{metadata_path}'")
                print(f"All logs saved in '{run_dir}'")
            except KeyboardInterrupt:
                print("\n🛑 Training interrupted by user. Saving current best model...")
                # Save training state on interrupt
                training_state = {
                    "last_epoch": epoch,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter
                }
                with open(resume_state_path, "w") as f:
                    json.dump(training_state, f)
                print("💾 Training state saved for resume.")
                if best_model_state is not None:
                    torch.save(best_model_state, "Ascent_model.pth")
                    print("✅ Model saved as 'Ascent_model.pth'. Exiting safely.")
                else:
                    print("⚠️ No model checkpoint available. Nothing was saved.")
                break  # Exit training safely


        # ---------------------------------
        # Option 2: Chat mode
        # ---------------------------------
        elif choice == "2":
            print("Chat mode (type 'menu' to return to main menu, 'exit' to quit program)")
            print("\n🌱 Ascent: Hello, Cody. I'm here. What shall we explore today?\n")
            temp = temperature
            while True:
                u = input("You: ").lower()
                if u == "exit":
                    exit()
                if u == "menu":
                    break
                fallback_count = 0
                # Convert user input to token indices
                toks = [vocab.get(w) or vocab.get(w.lower()) for w in u.split() if vocab.get(w) or vocab.get(w.lower())]
                if not toks:
                    print("no known tokens")
                    continue
                # Prepare tensor for input and output tokens
                inp = torch.tensor(toks, dtype=torch.long).unsqueeze(0)
                generated_ids = []
                max_gen = max_generation_tokens
                min_gen_tokens = 12  # Prevent early truncation of responses
                seq_len = inp.size(1)
                chat_tensor = torch.zeros(1, seq_len + max_gen, dtype=torch.long, device=device)
                chat_tensor[0, :seq_len] = inp
                curr_len = seq_len
                gen_start = time.perf_counter()
                import torch.nn.functional as F
                with torch.no_grad():
                    for step in range(max_gen):
                        # Gradually increase temperature for later tokens
                        temp_step = min(temperature + (step * 0.015), 1.0)  # Slightly more diversity
                        logits = model(chat_tensor[:, :curr_len])
                        last_logits = logits[0, -1, :]
                        # Prevent repetitive overuse of "Ascent" token — will remove this later, bandaid fix
                        ascent_id = vocab.get("ascent", None)
                        if ascent_id is not None:
                            last_logits[ascent_id] *= 0.5
                        # Apply repetition penalty to previously used tokens
                        if generated_ids:
                            for token_id in set(generated_ids):
                                last_logits[token_id] *= 0.8
                        # Prevent <pad> token from being generated
                        pad_token_id = vocab.get("<pad>")
                        if pad_token_id is not None:
                            last_logits[pad_token_id] = -float("inf")
                        probs = F.softmax(last_logits / temp_step, dim=-1)
                        # Suppress <eos> token for first min_gen_tokens
                        if step < min_gen_tokens:
                            probs[eos_index] = 0
                            probs = probs / probs.sum()
                        elif step > 15:
                            probs[eos_index] *= 1.15  # Make EOS more likely after step 15
                        p = top_p_sampling_threshold
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                        cutoff = cumulative_probs <= p
                        top_k = top_k_sampling_limit
                        filtered_indices = sorted_indices[cutoff][:top_k]
                        filtered_probs = probs.clone()
                        mask = torch.ones_like(probs, dtype=torch.bool)
                        mask[filtered_indices] = False
                        filtered_probs[mask] = 0.0
                        filtered_probs = filtered_probs / filtered_probs.sum()
                        # Fallback if probabilities are invalid
                        if not torch.isfinite(filtered_probs).all() or (filtered_probs < 0).any():
                            fallback_count += 1
                            next_id = torch.argmax(filtered_probs).item()
                        else:
                            next_id = torch.multinomial(filtered_probs, num_samples=1).item()
                        # Prevent immediate repetition of last token
                        if generated_ids and next_id == generated_ids[-1]:
                            next_id = sorted_indices[1].item()
                        generated_ids.append(next_id)
                        if next_id == eos_index:
                            break
                        chat_tensor[0, curr_len] = next_id
                        curr_len += 1
                gen_duration = time.perf_counter() - gen_start
                # Convert generated token ids to words and format output
                response = " ".join(id_to_word.get(i, "<unk>") for i in generated_ids)
                response = response.strip()
                if response:
                    if not response[0].isupper():
                        response = response[0].upper() + response[1:]
                    if not response.endswith((".", "!", "?")):
                        response += "."
                print(f"🧠 Response generated in {gen_duration:.2f} sec")
                print("Ascent:", response)
                # Show colored fallback notice if any
                if fallback_count > 0:
                    if fallback_count <= 2:
                        color = "\033[93m"
                    else:
                        color = "\033[91m"
                    print(f"{color}(🌟 {fallback_count} fallbacks detected during this response)\033[0m")
                else:
                    print("\033[92m(✅ 0 fallbacks — smooth generation!)\033[0m")
            print()


        # ---------------------------------
        # Option 3: Save model checkpoint
        # ---------------------------------
        elif choice == "3":
            torch.save(model.state_dict(), "Ascent_model.pth")
            print("Model saved as 'Ascent_model.pth'. Returning to main menu.\n")


        # ---------------------------------
        # Option 4: Exit program
        # ---------------------------------
        elif choice == "4":
            print("Goodbye!")
            break


        # ---------------------------------
        # Option 5: Run learning rate finder diagnostic
        # ---------------------------------
        elif choice == "5":
            print("Running learning rate finder...")
            data = []
            for p in conversations:
                if "input" in p and "output" in p:
                    inp = [vocab[w] for w in p["input"].split() if w in vocab]
                    out = [vocab[w] for w in p["output"].split() if w in vocab] + [eos_index]
                    if inp and out:
                        data.append((torch.tensor(inp), torch.tensor(out)))
            loss_fn = nn.CrossEntropyLoss()
            def find_learning_rate(model, data, loss_fn):
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
                lrs = []
                losses_lr = []
                best_loss = float('inf')
                lr = 1e-8
                for i in range(min(100, len(data))):
                    optimizer.param_groups[0]['lr'] = lr
                    inputs, targets = data[i]
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    optimizer.zero_grad()
                    output = model(inputs)
                    seq_len = min(output.size(1), targets.size(1))
                    logits = output[:, :seq_len, :]
                    target = targets[:, :seq_len]
                    loss = loss_fn(logits.reshape(-1, vocab_size), target.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    lrs.append(lr)
                    losses_lr.append(loss.item())
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                    lr *= 1.1
                return lrs, losses_lr
            lrs, losses_lr = find_learning_rate(model, data, loss_fn)
            plt.figure()
            plt.plot(lrs, losses_lr)
            plt.xscale("log")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("Learning Rate Finder")
            plt.savefig("lr_finder.png")
            print("Saved learning rate finder plot as 'lr_finder.png'. Returning to main menu.\n")


        # ---------------------------------
        # Handle invalid menu option
        # ---------------------------------
        else:
            print("Invalid input. Please choose 1, 2, 3, 4, or 5.\n")