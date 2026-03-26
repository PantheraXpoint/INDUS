import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the table
approaches = ['Vectorized\nRetrieval', 'AVA', 'VGENT', 'INDUS']

# LVBench Data (Latency removed)
lv_tokens = [8639, 23630, 13727, 12731]
lv_acc = [28.92, 33.96, 30.08, 37.03]

# AVA100 Data (Latency removed)
ava_tokens = [13923, 21822, 17474, 15954]
ava_acc = [30.00, 33.30, 39.17, 41.66]

# Set up the x-axis positions and bar width
x = np.arange(len(approaches))
width = 0.35  

# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

def plot_dataset(ax_primary, title, tok_data, acc_data):
    """Helper function to plot Tokens Used and Response Accuracy."""
    
    # 1. Plot Tokens Used (Left Y-Axis - Bar)
    color_tok = 'tab:blue'
    ax_primary.set_xlabel('Approach', fontweight='bold')
    ax_primary.set_ylabel('Tokens Used', color=color_tok, fontweight='bold')
    bars_tok = ax_primary.bar(x - width/2, tok_data, width, label='Tokens Used', color=color_tok)
    ax_primary.tick_params(axis='y', labelcolor=color_tok)
    ax_primary.set_xticks(x)
    
    # Set x-ticks and rotate slightly for better readability
    ax_primary.set_xticklabels(approaches, rotation=15, ha='center')
    ax_primary.set_title(title, fontweight='bold', fontsize=14, pad=15)
    
    # 2. Plot Response Accuracy (Right Y-Axis - Bar)
    ax_acc = ax_primary.twinx()
    color_acc = 'tab:orange'
    ax_acc.set_ylabel('Response Acc. (%)', color=color_acc, fontweight='bold')
    bars_acc = ax_acc.bar(x + width/2, acc_data, width, label='Response Acc. (%)', color=color_acc)
    ax_acc.tick_params(axis='y', labelcolor=color_acc)

    # Combine legends from both axes
    lines_1, labels_1 = ax_primary.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc.get_legend_handles_labels()
    
    # Place legend inside the plot
    ax_primary.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Plot LVBench on the left subplot
plot_dataset(ax1, 'LVBench', lv_tokens, lv_acc)

# Plot AVA100 on the right subplot
plot_dataset(ax2, 'AVA100', ava_tokens, ava_acc)

# Adjust layout so legends and labels don't overlap
plt.tight_layout()

# Save the plot
plt.savefig('visualization/response_efficiency_simplified.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Plot saved successfully as 'response_efficiency_simplified.png'")