# create_dataset.py
# Loads the Iris dataset from sklearn, exports it to CSV,
# and creates a scatter plot of petal length vs petal width colored by species.
# Consistent with model.py, PRD.md, REPORT.md and PLAN.md.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ─── 1. Load the Iris dataset ────────────────────────────────────────────────
iris = load_iris()
X = iris.data        # shape (150, 4)
y = iris.target      # 0=setosa, 1=versicolor, 2=virginica
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# ─── 2. Save to CSV ───────────────────────────────────────────────────────────
df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = [class_names[label] for label in y]

df.to_csv('dataset.csv', index=False)
print(f"Saved: dataset.csv  ({len(df)} rows, {len(df.columns)} columns)")

# ─── 3. Scatter plot: petal_length vs petal_width, colored by species ────────
# Colors match the three classes used throughout the project
colors  = ['#e74c3c', '#2ecc71', '#3498db']   # setosa=red, versicolor=green, virginica=blue
markers = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(7, 5))

for i, (cls, color, marker) in enumerate(zip(class_names, colors, markers)):
    mask = y == i
    ax.scatter(
        df.loc[mask, 'petal_length'],
        df.loc[mask, 'petal_width'],
        c=color,
        marker=marker,
        label=cls.capitalize(),
        edgecolors='white',
        linewidths=0.5,
        s=70,
        alpha=0.85
    )

ax.set_title('Iris Dataset - Petal Length vs Petal Width', fontsize=13, fontweight='bold')
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.legend(title='Species')
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('dataset_plot.png', dpi=150)
plt.close()
print("Saved: dataset_plot.png")
