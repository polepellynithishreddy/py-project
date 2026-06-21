# Anomaly Detection in Network/Financial Transactions Using SMOTE

A comparative study of **supervised vs. unsupervised** machine learning models for anomaly/fraud detection, including the effect of **SMOTE** (Synthetic Minority Over-sampling Technique) on handling class imbalance.

The full analysis lives in [`anomaly_detection_with_smote-final.ipynb`](./anomaly_detection_with_smote-final.ipynb).

## What this notebook does

1. **EDA** on two datasets — a bank marketing dataset and a synthetic credit-card transactions dataset (distributions, correlations, IQR outlier checks).
2. **Preprocessing** — label encoding, feature scaling (`StandardScaler`), and PCA (2D) for visualization.
3. **Unsupervised anomaly detection** — Isolation Forest.
4. **Supervised classification** — Random Forest, evaluated with classification report, confusion matrix, and ROC-AUC.
5. **Class imbalance handling** — SMOTE applied to the training set, with before/after comparison of precision, recall, F1, and ROC-AUC.
6. **Neural networks** — `MLPClassifier` (supervised) and an `MLPRegressor`-based autoencoder (unsupervised, anomaly score = reconstruction error), each with and without SMOTE.
7. **Hyperparameter tuning** — `RandomizedSearchCV` for Random Forest and MLP; manual grid search for Isolation Forest and the autoencoder.
8. **Final comparison** — a combined ROC-AUC chart across all 14 model variants (baseline/tuned × with/without SMOTE × supervised/unsupervised).

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `imbalanced-learn` (provides `SMOTE`)
- `jupyter` (to run the notebook)

Install everything with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

## Data

The notebook expects two CSV files:

| Variable | Description | Source |
|---|---|---|
| `bank_path` | "Bank Marketing" dataset (`bank-full.csv`), with a binary `class` column (`yes`/`no`) used as the supervised target. | [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| `credit_path` | A synthetic credit-card transactions dataset (`synthetic_card_transactions.csv`), used unlabeled — only the `Amount` column is used for anomaly detection. | [Mendeley Data — Chapter 14: Techniques for Detecting Fraud — Synthetic Credit Card Transaction Dataset](https://data.mendeley.com/datasets/wydfx7yhcx/1) |

**Before running:** update the hardcoded file paths in the *"Data Loading"* cell (near the top of the notebook) to point to your own local copies of these files:

```python
bank_path   = 'path/to/bank-full.csv'
credit_path = 'path/to/synthetic_card_transactions.csv'
```

## How to run

```bash
git clone <this-repo-url>
cd <this-repo>
jupyter notebook anomaly_detection_with_smote-final.ipynb
```

Run all cells top-to-bottom — later sections (SMOTE, neural networks, hyperparameter tuning) depend on variables created earlier (`X_train`, `y_train`, `X_test`, `y_test`, `rf_auc`, etc.), so the notebook is not designed to be run out of order.

Each major chart is also saved to disk as a `.png` in the working directory (e.g. `bank_distributions.png`, `confusion_matrix_smote.png`, `final_model_comparison_all.png`) for easy reuse in reports or slides.

## Notes for reuse

- **Runtime:** the hyperparameter tuning section (Section 12) is the slowest part of the notebook — it fits dozens of model variants. Comment it out or reduce `n_iter` / grid sizes if you just want the core EDA → model → SMOTE comparison.
- **Reproducibility:** all stochastic steps (`train_test_split`, models, SMOTE) use `random_state=42`.
- **Swapping in your own data:** as long as your dataset has a binary target column, you can substitute it for the bank dataset by updating the column name in the preprocessing cell (currently `class`).
- **Metrics:** ROC-AUC is used throughout for comparability between supervised and unsupervised methods. For the unsupervised models, the anomaly score (Isolation Forest's decision function, or autoencoder reconstruction error) is treated as the predicted probability when computing ROC-AUC.

## License

Add a license of your choice (e.g. MIT) here if you intend others to reuse this code.
