import matplotlib.pyplot as plt

# X-axis data: Time points
time_points = [400, 600, 800, 1000, 1200]

# Y-axis data: Hit rates for each approach (%)
# Extracted from the table
vcr_hit_rate = [31.82, 34.55, 36.36, 38.18, 38.18]
ava_hit_rate = [38.00, 44.00, 48.00, 50.00, 51.80]
indus_hit_rate = [46.36, 55.45, 61.82, 62.73, 64.55]

# Create the figure
plt.figure(figsize=(10, 6))

# Plot the data lines with distinct markers
plt.plot(time_points, vcr_hit_rate, marker='o', linestyle='-', linewidth=2, label='Vectorized Caption Retrieval', color='tab:blue')
plt.plot(time_points, ava_hit_rate, marker='s', linestyle='-', linewidth=2, label='AVA', color='tab:orange')
plt.plot(time_points, indus_hit_rate, marker='^', linestyle='-', linewidth=2, label='INDUS', color='tab:green')

# Add titles and labels
plt.title('Hit Rate Over Time on AVA100 (10-hour+ video)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Time', fontweight='bold', fontsize=12)
plt.ylabel('Hit Rate (%)', fontweight='bold', fontsize=12)

# Ensure the x-axis only shows the specific time points
plt.xticks(time_points)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Add the legend
plt.legend(loc='lower right', fontsize=11)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the plot to an image file
plt.savefig('visualization/hit_rate_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved successfully as 'hit_rate_over_time.png'")