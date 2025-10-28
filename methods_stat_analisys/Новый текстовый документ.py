import matplotlib.pyplot as plt
import numpy as np

# Sample numerical data
data = np.random.normal(50, 10, 1000) # 1000 data points from a normal distribution

# Define the bins (intervals)
bins = np.arange(20, 90, 5) # Bins from 20 to 85 with a step of 5

# Calculate the frequency of data in each bin
counts, _ = np.histogram(data, bins=bins)

# Create labels for the intervals
interval_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)]

plt.figure(figsize=(10, 6))
plt.bar(interval_labels, counts, color='lightgreen', edgecolor='black')
plt.xlabel('Value Intervals')
plt.ylabel('Frequency')
plt.title('Frequency Bar Chart by Intervals')
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.tight_layout()
plt.show()
