import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Resolve paths
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Configuration parameters
target_molecule = "H2O"
USE_ECP_AVAS = True
seq_len = 4
block_size = 256

setting_suffix = ""
if USE_ECP_AVAS:
    setting_suffix += "_ecpavas"

trial_name = f"trial_{target_molecule.lower()}{setting_suffix}"
save_dir = os.path.join(current_file_dir, "data", f"seq_len={seq_len}", trial_name)

# Check for both model paths (live output and backup)
model_path = os.path.join(save_dir, "gqe.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(save_dir, "gqe.pt.bak")

print(f"Target Sequence Length: {seq_len} tokens")
print(f"Transformer Block Size: {block_size} tokens")

# 2. Check if a pre-trained model exists to load the actual sequence
best_seq_tokens = [0] * seq_len
if os.path.exists(model_path):
    try:
        # Load the model and retrieve details
        gpt = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"Successfully loaded pre-trained GQE model from: {model_path}")
        
        # Load cached true energies to find the best sequence (check both csv and bak)
        true_Es_path = os.path.join(save_dir, "true_Es_t.csv")
        if not os.path.exists(true_Es_path):
            true_Es_path = os.path.join(save_dir, "true_Es_t.bak")
        
        if os.path.exists(true_Es_path):
            import pandas as pd
            df_true = pd.read_csv(true_Es_path).iloc[:, 1:]
            last_iteration_column = df_true.columns[-1]
            true_Es = df_true[last_iteration_column].values
            
            # Retrieve the corresponding generated sequences
            best_idx = np.argmin(true_Es)
            print(f"Found best ansatz sequence at index {best_idx} with energy {true_Es[best_idx]:.5f} Ha")
    except Exception as e:
        print(f"Could not load best sequence details: {e}. Using default dummy tokens.")

# 3. Create visualization plot
fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)

# Draw the 256-token block context window
ax.barh(0, block_size, align='center', height=0.4, color='#e2e8f0', label='Unused Context Capacity', edgecolor='#cbd5e1')

# Draw the GQE ansatz sequence inside the block
ax.barh(0, seq_len, align='center', height=0.4, color='#3b82f6', label='GQE Ansatz Sequence', edgecolor='#1d4ed8')

# Decorate plot
ax.set_yticks([])
ax.set_xlim(-10, block_size + 10)
ax.set_xlabel('Token Index / Position in Context Window', fontsize=12)
ax.set_title(f'GQE Ansatz Sequence Length ({seq_len} gates) vs. Transformer Block Size ({block_size} tokens)', fontsize=13, fontweight='bold', pad=15)

# Annotate values
ax.text(seq_len / 2 + 1, 0, f'{seq_len} tokens ({(seq_len/block_size)*100:.1f}%)', 
        ha='left', va='center', color='#1e3a8a', fontweight='bold', fontsize=10)
ax.text(block_size - (block_size - seq_len) / 2, 0, f'{block_size - seq_len} tokens free', 
        ha='center', va='center', color='#475569', fontsize=10)

# Grid and legend
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2, frameon=True)

# Save the visualization plot
output_img_path = os.path.join(current_file_dir, "gqe_sequence_block_fit.png")
plt.tight_layout()
plt.savefig(output_img_path)
print(f"Visualization saved to {output_img_path}")
