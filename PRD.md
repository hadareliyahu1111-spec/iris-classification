# Product Requirements Document: Iris Flower Classifier

## Overview

Build a machine learning classifier for the Iris flower dataset that trains a model, evaluates its performance via a Confusion Matrix, visualizes the training loss curve, and saves all generated graphs as image files.

---

## Goals

- Load the standard Iris dataset from `sklearn.datasets`
- Split data into 80% training / 20% testing sets
- Train a classification model
- Display and save a Confusion Matrix
- Plot and save a loss curve over training iterations

---

## Functional Requirements

### 1. Dataset

- **Source:** `sklearn.datasets.load_iris()`
- **Features:** 4 numeric features (sepal length, sepal width, petal length, petal width)
- **Target:** 3 classes — Setosa, Versicolor, Virginica

### 2. Data Split

- Use `train_test_split` from `sklearn.model_selection`
- Split ratio: **80% train / 20% test**
- Set a fixed `random_state` for reproducibility

### 3. Model

- Train a classification model (e.g., MLPClassifier, Logistic Regression, or SVM)
- The model must expose per-iteration loss values to enable a loss curve (e.g., `MLPClassifier` with `max_iter` set and `verbose=False`)
- Record loss at each training iteration

### 4. Confusion Matrix

- Compute predictions on the test set
- Generate a Confusion Matrix using `sklearn.metrics.confusion_matrix`
- Display it as a heatmap (using `seaborn.heatmap` or `matplotlib`)
- Label axes with class names
- Save as **`confusion_matrix.png`**

### 5. Loss Curve

- Plot the model's loss value on the y-axis against iteration number on the x-axis
- Title: "Training Loss Curve"
- Label axes: "Iteration" (x), "Loss" (y)
- Save as **`loss_curve.png`**

### 6. Output Files

| File                  | Description                          |
|-----------------------|--------------------------------------|
| `confusion_matrix.png`| Heatmap of predicted vs. actual labels |
| `loss_curve.png`      | Loss value plotted over training iterations |

---

## Non-Functional Requirements

- **Language:** Python 3.8+
- **Dependencies:** `scikit-learn`, `matplotlib`, `seaborn`, `numpy`
- **Reproducibility:** Fixed random seeds throughout
- **Code quality:** Single self-contained script (`classifier.py`)

---

## Acceptance Criteria

- [ ] Script runs end-to-end without errors
- [ ] Dataset is loaded from sklearn (no external files)
- [ ] Data is split 80/20 with a fixed random seed
- [ ] A classification model is trained
- [ ] `confusion_matrix.png` is generated and saved
- [ ] `loss_curve.png` is generated and saved
- [ ] Both images are readable and correctly labeled

---

## Out of Scope

- Hyperparameter tuning or cross-validation
- Deployment or serving of the model
- Interactive dashboards or web UI
- Additional datasets beyond Iris
