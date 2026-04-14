# model.py
# Iris Flower Classifier — 4 classes (setosa, versicolor, virginica, iris-fake)
# Trains an MLPClassifier on the combined dataset, evaluates it,
# and saves confusion_matrix.png and loss_curve.png.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_iris

# ─── 1. Rebuild the 4-class dataset (same logic as create_dataset.py) ────────
iris = load_iris()
X_real = iris.data
y_real = iris.target

rng = np.random.default_rng(seed=42)
mean_features = X_real.mean(axis=0)
std_features  = X_real.std(axis=0) * 0.5
X_fake = rng.normal(loc=mean_features, scale=std_features, size=(50, 4))
X_fake = np.clip(X_fake, 0.1, None)

X = np.vstack([X_real, X_fake])
y = np.concatenate([y_real, np.full(50, 3, dtype=int)])

class_names = ['Setosa', 'Versicolor', 'Virginica', 'Iris-Fake']

# ─── 2. Split data 80% training / 20% testing ────────────────────────────────
# stratify=y keeps class proportions equal in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─── 3. Train the MLPClassifier ──────────────────────────────────────────────
# Two hidden layers (64 → 32 neurons), ReLU activation, Adam optimiser
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)
print(f"\nTraining finished after {model.n_iter_} iterations.")

# ─── 4. Predict and print the Confusion Matrix ───────────────────────────────
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ─── 5. Calculate and print Accuracy ─────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ─── 6. Save confusion_matrix.png (4x4) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax
)
ax.set_title('Confusion Matrix — 4-Class Iris Classifier', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ─── 7. Save loss_curve.png ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(model.loss_curve_, color='steelblue', linewidth=2)
ax.set_title('Training Loss Curve — 4-Class Iris Classifier', fontsize=13, fontweight='bold')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.close()
print("Saved: loss_curve.png")
