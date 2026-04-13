# model.py
# Iris Flower Classifier using MLPClassifier
# Trains a neural network, evaluates it, and saves plots.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ─── 1. Load the Iris dataset ────────────────────────────────────────────────
iris = load_iris()
X = iris.data        # Feature matrix: (150, 4)
y = iris.target      # Labels: 0=Setosa, 1=Versicolor, 2=Virginica
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# ─── 2. Split data 80% training / 20% testing ────────────────────────────────
# stratify=y keeps the same class proportions in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─── 3. Train the MLPClassifier ──────────────────────────────────────────────
# Two hidden layers (64 → 32 neurons), ReLU activation, Adam optimiser.
# max_iter=300 lets the loss_curve_ accumulate enough points to plot.
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)
print(f"\nTraining converged after {model.n_iter_} iterations.")

# ─── 4. Predict and print the Confusion Matrix ───────────────────────────────
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ─── 5. Calculate and print Accuracy ─────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ─── 6. Save confusion_matrix.png ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,          # write counts inside each cell
    fmt='d',             # integer format
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax
)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ─── 7. Save loss_curve.png ──────────────────────────────────────────────────
# MLPClassifier stores the loss at every iteration in loss_curve_
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(model.loss_curve_, color='steelblue', linewidth=2)
ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.close()
print("Saved: loss_curve.png")
