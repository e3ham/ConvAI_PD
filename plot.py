import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Whisper-base", "Whisper-base (weighted)", "Whisper-medium (weighted)", "Whisper-large (weighted)"]

# Accuracy values
overall_acc = [0.6221, 0.6860, 0.8140, 0.7616]
hc_acc = [0.9333, 0.6167, 0.75, 0.80]
pd_acc = [0.4554, 0.7232, 0.85, 0.74]

# X-axis locations
x = np.arange(len(models))
width = 0.25

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars1 = ax.bar(x - width, overall_acc, width, label='Overall Accuracy')
bars2 = ax.bar(x, hc_acc, width, label='HC Accuracy')
bars3 = ax.bar(x + width, pd_acc, width, label='PD Accuracy')

# Add labels and title
ax.set_ylabel('Accuracy')
ax.set_xlabel('Model Variant')
ax.set_title('Accuracy by Model Variant and Group')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10)
ax.set_ylim(0, 1.1)
ax.legend()

# Show plot
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
plt.savefig("model_accuracies.png", dpi=300)
