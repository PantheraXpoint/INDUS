import matplotlib.pyplot as plt
import numpy as np

# Updated x-axis labels without the newline
approaches = [
    'w/out verifier', 
    'event-based', 
    'object-based', 
    'full'
]

# Re-using the exact data
hit_rate = [46.36, 53.64, 58.18, 61.82]
cov_rate = [40.33, 49.92, 54.81, 58.64]

# Set up the x-axis positions and bar width
x = np.arange(len(approaches))
width = 0.35  

# Keep the base figure spacious to accommodate massive fonts
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the grouped bars for Accuracy using the specific hex colors requested
color_hit = '#4C78A8'  # Custom Blue
color_cov = '#59A14F'  # Custom Green
bars_hit = ax1.bar(x - width/2, hit_rate, width, label='Hit Rate (%)', color=color_hit)
bars_cov = ax1.bar(x + width/2, cov_rate, width, label='Coverage Rate (%)', color=color_cov)

# EXTRA LARGE fonts for axis labels and title to match table sizes
ax1.set_xlabel('Approach', fontweight='bold', fontsize=20)
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=20)

# Set y-axis limit higher to leave room for the large text labels on top
ax1.set_ylim(0, max(max(hit_rate), max(cov_rate)) + 15)

# --- Define Text Annotation Font ---
annotation_font = {'fontsize': 16, 'fontweight': 'bold', 'color': 'black'}

# Add text labels on top of the bars with EXTRA LARGE, bold fonts
for bar in bars_hit:
    yval = bar.get_height()
    # Blue bar values are centered as normal
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval}%', ha='center', va='bottom', **annotation_font)

# Move Green Bar Labels Right to prevent overlap
for bar in bars_cov:
    yval = bar.get_height()
    # Calculate new position shifted slightly to the right of the center
    x_pos = bar.get_x() + bar.get_width()/2 + 0.025 
    
    # Text is centered relative to the new right-shifted point
    ax1.text(x_pos, yval + 1.5, f'{yval}%', ha='center', va='bottom', **annotation_font)

# Set the x-axis labels with EXTRA LARGE fonts
ax1.set_xticks(x)
ax1.set_xticklabels(approaches, rotation=0, ha='center', fontsize=18)
ax1.tick_params(axis='y', labelsize=18)

# Add a title with an EXTRA LARGE font
plt.title('Accuracy: Hit Rate vs Coverage Rate', fontsize=24, fontweight='bold', pad=20)

# Add legend with an EXTRA LARGE font
ax1.legend(loc='upper left', framealpha=0.9, fontsize=16)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Save the plot with high DPI for crisp scaling
plt.savefig('visualization/accuracy_hit_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved successfully as 'accuracy_hit_coverage_single_line.png'")