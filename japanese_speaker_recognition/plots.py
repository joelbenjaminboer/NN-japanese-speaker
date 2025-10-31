import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
file_path = "data/ae.test"  # change to ae.test if desired
n_features = 12

# === LOAD AND SPLIT INTO BLOCKS ===
utterances = []
block = []

with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            # for .test with blank lines
            if block:
                utterances.append(np.array(block, dtype=float))
                block = []
            continue

        values = np.array(list(map(float, line.split())))
        # separator line = all 1.0
        if np.allclose(values, 1.0, atol=1e-6):
            if block:
                utterances.append(np.array(block, dtype=float))
                block = []
        else:
            block.append(values)

# add last utterance
if block:
    utterances.append(np.array(block, dtype=float))

print(f"Loaded {len(utterances)} utterances from {file_path}")

# === 1️⃣ UTTERANCE-LEVEL CORRELATION (mean per utterance) ===
mean_vectors = [np.mean(u, axis=0) for u in utterances]
df_utterance = pd.DataFrame(mean_vectors, columns=[f"LPC{i+1}" for i in range(n_features)])
corr_utterance = df_utterance.corr()

# === 2️⃣ FRAME-LEVEL CORRELATION (all frames combined) ===
all_frames = np.vstack(utterances)
df_frame = pd.DataFrame(all_frames, columns=[f"LPC{i+1}" for i in range(n_features)])
corr_frame = df_frame.corr()

# === PLOTS ===
plt.figure(figsize=(18, 7))

# Utterance-level
plt.subplot(1, 2, 1)
sns.heatmap(
    corr_utterance, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'label': 'Correlation'}
)
plt.title(f"Utterance-level correlation ({file_path})", fontsize=13)

# Frame-level
plt.subplot(1, 2, 2)
sns.heatmap(
    corr_frame, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'label': 'Correlation'}
)
plt.title(f"Frame-level correlation ({file_path})", fontsize=13)

plt.tight_layout()
plt.savefig(f"frame_level_correlation.png", dpi=300)
