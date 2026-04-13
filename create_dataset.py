# create_dataset.py
# Exports the Iris dataset to CSV and creates a scatter plot colored by class.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ─── 1. Load the Iris dataset ────────────────────────────────────────────────
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
feature_names = iris.feature_names

# ─── 2. Export to CSV ────────────────────────────────────────────────────────
# Build a DataFrame with feature columns + a species label column
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [class_names[label] for label in y]

df.to_csv('dataset.csv', index=False)
print(f"Saved: dataset.csv  ({len(df)} rows, {len(df.columns)} columns)")

# ─── 3. Scatter plot colored by class ────────────────────────────────────────
# Plot petal length vs petal width — the most discriminative feature pair
colors = ['#e74c3c', '#2ecc71', '#3498db']  # red, green, blue
markers = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(7, 5))

for i, (cls, color, marker) in enumerate(zip(class_names, colors, markers)):
    mask = y == i
    ax.scatter(
        X[mask, 2],   # petal length
        X[mask, 3],   # petal width
        c=color,
        marker=marker,
        label=cls.capitalize(),
        edgecolors='white',
        linewidths=0.5,
        s=70,
        alpha=0.85
    )

ax.set_title('Iris Dataset — Petal Length vs Petal Width', fontsize=13, fontweight='bold')
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.legend(title='Species')
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('dataset_plot.png', dpi=150)
plt.close()
print("Saved: dataset_plot.png")
