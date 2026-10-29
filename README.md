# 🩺 Liver Patient Classification Using ML Methods

Binary classification of patients as **liver patients** or **healthy subjects** from clinical and demographic data, developed for a [Kaggle InClass](https://www.kaggle.com/t/1ccb38a0e7054e6dbe2c953c10cf0124) competition. The work covers the full ML cycle: exploratory analysis, preprocessing, model comparison, hyperparameter search, ensembles and handling of class imbalance.

> Project for *Machine Learning 1* (AA1) · Universitat Politècnica de Catalunya (UPC) · Bachelor's in Data Science and Engineering (GCED) · Academic year 2024–2025
> Authors: **Lucas Viner** and **Oriol Parent**

---

## 📌 Overview

The goal is to build a classifier that predicts whether a subject is ill (liver patient) or healthy from 10 features. The competition metric is the **F1-score**, and the dataset is imbalanced, so the whole pipeline is designed around that.

**Dataset:** Indian Liver Patient Dataset (ILPD, UCI ML Repository), preprocessed by the course staff (no missing values, categorical features encoded numerically).

| File | Content |
|---|---|
| `train_features_ILDS.csv` | 463 × 10 training features |
| `train_labels_ILDS.csv` | 463 training labels |
| `test_data_ILDS.csv` | 116 × 10 test features (labels hidden, scored by Kaggle) |

**Features:** `Age`, `Female`, `TB` (total bilirubin), `DB` (direct bilirubin), `Alkphos`, `Sgpt`, `Sgot`, `TP` (total proteins), `ALB` (albumin), `A/R` (albumin/globulin ratio).

## 🔬 Methodology

1. **Data split:** stratified 80/20 split of the training set into a training set and an *external validation* set (the Kaggle test set is used only for the final submission).
2. **Exploratory analysis and preprocessing**
   - No missing values found.
   - **Outliers were kept**: they are clinically meaningful, and removing them lowered the F1-score.
   - **Log transformation** (`log10`) of skewed variables to approximate normality and unify ranges.
   - **Collinearity:** `TB` and `DB` are highly correlated, so `DB` is dropped.
   - Everything is wrapped in a `preprocess_features` function used inside a scikit-learn `FunctionTransformer`, so the same transformations apply in training, cross-validation and inference (no data leakage).
3. **Class imbalance:** `class_weight='balanced'` / `'balanced_subsample'` explored as a hyperparameter. **SMOTE** was also tested but gave no improvement.
4. **Models compared:** Logistic Regression (with and without L1), SVM (linear and RBF), KNN, Decision Tree, Random Forest, Extra Trees, Bagging + LR, LDA, QDA and Naive Bayes.
5. **Hyperparameter search:** `GridSearchCV` over a pipeline `preprocessing → scaler → PCA → model`, searching scaler type (Standard / Robust / MinMax / none), PCA variance retained and model-specific parameters. Evaluation with `RepeatedStratifiedKFold` (5 folds × 10 repeats), optimising **F1 macro**.
6. **Ensembles:** hard and soft `VotingClassifier` (with weight search) and `StackingClassifier` with several meta-learners.
7. **Final model:** refit on the full training data (train + external validation) and predictions generated for the test set.

## 📊 Metrics

Accuracy can be misleading with imbalanced classes, so the following were tracked: accuracy, precision, recall, F1 per class (0 and 1) and **F1 macro** (main selection criterion, equal weight for both classes).

## 🏆 Key results

Cross-validated F1 macro of the best simple model per family:

| Model | F1 macro (CV) |
|---|---|
| **Extra Trees** | **0.682** |
| Random Forest | 0.647 |
| Logistic Regression (L1) | 0.645 |
| Logistic Regression | 0.642 |
| Bagging + LR | 0.639 |
| Decision Tree | 0.634 |
| SVM (RBF) | 0.627 |
| KNN | 0.622 |
| SVM (Linear) | 0.609 |
| QDA | 0.602 |
| Naive Bayes | 0.566 |
| LDA | 0.536 |

- Tree-based models performed best, suggesting non-linear relationships between variables.
- **No ensemble (voting or stacking) outperformed the best simple model**, so the final solution is a tuned **Extra Trees** classifier, which is simpler and cheaper to run.
- Scaling and PCA clearly helped distance-based models (KNN, SVM), but not tree models.

## 📁 Repository structure

```
.
├── code.ipynb                   # Full pipeline: EDA, preprocessing, models, ensembles, submission
├── Report.pdf                   # Final report (IEEE format, in Catalan)
├── AA1_CompetitionGuide_P25.pdf # Competition statement
├── train_features_ILDS.csv      # Training features (provided by the course)
├── train_labels_ILDS.csv        # Training labels
└── test_data_ILDS.csv           # Test features
```

> The notebook reads the three CSV files from its own folder, so they must be placed next to `code.ipynb`.

## ▶️ How to reproduce

**Requirements:** Python 3.9+ and the following packages:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn jupyter
```

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Add the three dataset CSV files to the folder.
3. Run `code.ipynb` from top to bottom:
   ```bash
   jupyter notebook code.ipynb
   ```
   The grid searches use repeated cross-validation and can take a while. `n_jobs=-1` uses all available cores.
4. The last cell writes `submission.csv` (116 rows, columns `ID`, `Label`, where 0 = liver patient and 1 = healthy).

## 🛠️ Tools

- **Python** · pandas, NumPy, SciPy
- **scikit-learn** · pipelines, `GridSearchCV`, `RepeatedStratifiedKFold`, ensembles
- **imbalanced-learn** · SMOTE and `ImbPipeline`
- **matplotlib / seaborn** · exploratory visualisation

## 📚 References

- Lichman, M. (2013). *UCI Machine Learning Repository*. University of California, Irvine. http://archive.ics.uci.edu/ml
- Course notes and lab sessions, *Aprenentatge Automàtic 1*, UPC, 2024–2025.

## 👥 Authors

- **Lucas Viner**
- **Oriol Parent**
