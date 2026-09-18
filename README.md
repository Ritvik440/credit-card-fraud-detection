# Credit Card Fraud Detection: Classification & Imbalance Benchmarking

An end-to-end Machine Learning and Data Analytics study evaluating classification models, class imbalance mitigation techniques, and decision-threshold trade-offs on highly skewed credit card transaction data.

---

## 1. Project Overview & Problem Statement

Credit card fraud represents an asymmetric risk problem: fraudulent transactions are rare but cause severe direct financial losses, while legitimate transactions form the vast majority of volume. Automated fraud classification models must accurately flag fraudulent activity without overwhelming operational review teams with false alarms or creating unnecessary friction for legitimate cardholders.

This project investigates binary fraud classification on the standard Kaggle European cardholder dataset. It establishes a leak-free machine learning workflow, examines class separation across latent PCA features, benchmarks three distinct classifier families across three class-imbalance paradigms, and evaluates decision-threshold dynamics.

---

## 2. Dataset Overview

- **Source**: Kaggle European cardholder dataset (September 2013).
- **Total Transactions**: 284,807 transactions recorded over a 48-hour period.
- **Class Breakdown**:
  - **Legitimate (Class 0)**: 284,315 transactions (99.828%)
  - **Fraudulent (Class 1)**: 492 transactions (0.172%)
  - **Imbalance Ratio**: Approximately 1 fraud per 578 legitimate transactions.
- **Features**:
  - `V1` to `V28`: 28 numerical features obtained through Principal Component Analysis (PCA) due to confidentiality constraints.
  - `Time`: Elapsed seconds between each transaction and the first transaction in the dataset.
  - `Amount`: Transaction monetary amount in euros.
  - `Class`: Response variable (1 for fraud, 0 for legitimate).

---

## 3. Key Challenge: Extreme Class Imbalance

With fraud occurring in only 0.172% of transactions, standard machine learning pipelines face specific pitfalls:
- **Trivial Accuracy**: A naive classifier predicting all transactions as legitimate achieves **99.828% accuracy**, rendering accuracy meaningless.
- **ROC-AUC Distortion**: Standard ROC-AUC curves can appear deceptively high because the enormous pool of true negatives ($>71,000$ in testing) compresses the False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$).
- **Need for Precision-Recall Rigor**: The **Area Under the Precision-Recall Curve (PR-AUC / Average Precision)** is the primary evaluation metric because it focuses strictly on the minority class trade-off between caught frauds (Recall) and false alarms (Precision).

---

## 4. Technologies Used

- **Language**: Python 3.12
- **Data Manipulation & Analysis**: NumPy (2.5.3), Pandas (3.0.6)
- **Machine Learning**: Scikit-Learn (1.9.1), XGBoost (3.4.1)
- **Class Imbalance Handling**: Imbalanced-Learn (`imblearn` 0.14.2)
- **Data Visualization**: Matplotlib (3.11.2), Seaborn (0.13.2)
- **Environment**: Jupyter Notebook

---

## 5. Exploratory Data Analysis (EDA)

The exploratory analysis focuses on class-conditioned distributions rather than unsegmented population averages:

1. **Class Distribution Quantification**:
   - Visualized target class distribution on a logarithmic scale, highlighting the 1:578 imbalance ratio.
2. **Transaction Amount Comparison**:
   - **Legitimate**: Mean = \$88.29, Median = \$22.00, 75th Percentile = \$77.05, Max = \$25,691.16.
   - **Fraudulent**: Mean = \$122.21, Median = \$9.25, 75th Percentile = \$105.89, Max = \$2,125.87.
   - Fraudulent transactions exhibit a smaller median amount but frequent clusters around specific values, while normal transactions exhibit a heavy right-tail of large purchases.
3. **Temporal Dynamics (`Hour_of_Day`)**:
   - Converted cumulative `Time` into cyclical 24-hour intervals (`Hour`).
   - Normal transactions follow typical diurnal activity, dropping substantially during night hours.
   - Fraud transactions show elevated relative density during off-peak and early morning hours (00:00 to 06:00).
4. **Target Correlation & Discriminative Feature Divergence**:
   - Identified the features most correlated with fraud:
     - Negative correlation: $V_{17}$ (-0.326), $V_{14}$ (-0.303), $V_{12}$ (-0.261), $V_{10}$ (-0.217).
     - Positive correlation: $V_{11}$ (+0.155), $V_4$ (+0.134).
   - Overlaid KDE density plots confirmed substantial class separation on $V_{14}, V_{17}, V_{12},$ and $V_{10}$.

---

## 6. Feature Engineering & Preprocessing

### Empirical Feature Engineering
- **`Hour` Extraction**: Derived $Hour = (\text{Time} // 3600) \pmod{24}$ to capture diurnal timing.
- **Pruning Raw Time**: Dropped raw cumulative seconds (`Time`) to eliminate dependence on an arbitrary recording start epoch.

### Methodological Preprocessing (Zero Data Leakage)
1. **Split Prior to Scaling**:
   - Split dataset into **75% Training** ($N = 213,605$, 369 frauds) and **25% Testing** ($N = 71,202$, 123 frauds).
   - Enforced stratified sampling (`stratify=y`) with a fixed seed (`random_state=42`) to guarantee identical class ratios across splits.
2. **Train-Only Parameter Fitting**:
   - `RobustScaler` was fitted strictly on `X_train` and then used to transform both `X_train` and `X_test`.
   - Guaranteed zero parameter leakage from the test distribution.

---

## 7. Model Architectures & Imbalance Handling Strategies

### Model Architectures
- **Logistic Regression**: Linear baseline providing interpretable log-odds coefficients and rapid inference.
- **Random Forest**: Bagging ensemble (100 estimators) capturing non-linear interactions and resilient to outliers.
- **XGBoost**: Gradient-boosted decision trees (GBDT, 100 estimators) sequentially optimizing logistic loss.

### Imbalance Handling Paradigms
1. **Baseline**: Trained directly on raw, unweighted training data.
2. **Cost-Sensitive / Class-Weighted**: Algorithmic penalty weighting:
   - Logistic Regression: `class_weight='balanced'`
   - Random Forest: `class_weight='balanced'`
   - XGBoost: `scale_pos_weight = 577.88` (computed from training negative-to-positive ratio)
3. **SMOTE (Synthetic Minority Oversampling)**:
   - Applied **strictly to the training fold** (`X_train_scaled`, rebalancing to 213,236 frauds vs. 213,236 legitimate samples).
   - The test set remained completely unmodified (0.172% fraud rate) to preserve valid real-world evaluation.

---

## 8. Evaluation Metrics

Because the test partition contains 71,079 legitimate cases and only 123 frauds, models were evaluated on:
- **Precision**: Proportion of flagged transactions that were actually fraudulent ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
- **Recall**: Proportion of actual fraudulent transactions successfully identified ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).
- **F1-Score**: Harmonic mean of Precision and Recall.
- **PR-AUC (Average Precision)**: Summary metric measuring precision across all recall thresholds; primary metric for imbalanced data.
- **ROC-AUC**: Standard discrimination metric across all false-positive rate thresholds.

---

## 9. Experimental Benchmark Results

All 9 configurations were evaluated on the identical unseen test set ($N = 71,202$ total, $123$ frauds):

| Rank | Model | Imbalance Strategy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **XGBoost** | **SMOTE** | **0.8175** | **0.8374** | **0.8273** | **0.8646** | **0.9823** |
| 2 | **XGBoost** | **Class-Weighted** | 0.8839 | 0.8049 | 0.8426 | 0.8617 | 0.9798 |
| 3 | **Random Forest** | **SMOTE** | 0.8772 | 0.8130 | 0.8439 | 0.8564 | 0.9660 |
| 4 | **Random Forest** | **Class-Weighted** | 0.9314 | 0.7724 | 0.8444 | 0.8476 | 0.9610 |
| 5 | **Random Forest** | **Baseline** | 0.9515 | 0.7967 | 0.8673 | 0.8437 | 0.9415 |
| 6 | **Logistic Regression** | **Baseline** | 0.8462 | 0.6260 | 0.7196 | 0.7205 | 0.9635 |
| 7 | **Logistic Regression** | **SMOTE** | 0.0569 | 0.8943 | 0.1070 | 0.7120 | 0.9734 |
| 8 | **Logistic Regression** | **Class-Weighted** | 0.0597 | 0.8862 | 0.1119 | 0.7055 | 0.9736 |
| 9 | **XGBoost** | **Baseline** | 0.1211 | 0.1870 | 0.1470 | 0.0531 | 0.4301 |

---

## 10. Key Empirical Findings

1. **Tree-Based Resilience & Separation**:
   - Both **XGBoost** and **Random Forest** achieved strong minority-class separation, recording PR-AUC scores between **0.843** and **0.865**.
2. **The SMOTE / Weighting Dilemma on Linear Models**:
   - For **Logistic Regression**, applying SMOTE or Class Weighting increased recall from $62.6\%$ to $\sim 89\%$, but caused a **severe collapse in precision** (dropping from $84.6\%$ down to $5.7\% - 6.0\%$). In an operational setting, this produces approximately 16 false alarms for every confirmed fraud case.
3. **Empirical Behavior of Baseline XGBoost**:
   - The unweighted default XGBoost configuration performed poorly on this dataset (PR-AUC 0.0531, ROC-AUC 0.4301) under the extreme 0.172% fraud rate. Diagnostic analysis indicated unstable tree-weight updates with the default unconstrained max_delta_step=0, while class weighting and SMOTE produced substantially stronger results.
4. **Primary Fraud Risk Predictors**:
   - Feature importance from the selected XGBoost + SMOTE model indicated that fraud predictions were predominantly driven by latent components:
     1. $V_{14}$ ($60.65\%$)
     2. $V_4$ ($6.60\%$)
     3. $V_{12}$ ($2.90\%$)
     4. $V_{10}$ ($2.68\%$)
     5. $V_8$ ($2.43\%$)

---

## 11. Confusion Matrix & Decision-Threshold Analysis

### Operational Confusion Matrix (Top Model: XGBoost + SMOTE)
At the default $0.50$ decision threshold on the test set:
- **True Negatives (Legitimate Cleared)**: 71,056 / 71,079
- **False Positives (False Alarms)**: 23
- **False Negatives (Missed Frauds)**: 20
- **True Positives (Frauds Prevented)**: 103 / 123 ($83.7\%$ fraud capture rate)

### Analytical vs. Operational Threshold Selection
- A sweep across classification thresholds from $0.05$ to $0.95$ identified **$0.81$** as the analytical threshold that maximizes F1-Score:
  - **At Threshold 0.81**: Precision = $0.9182$, Recall = $0.8211$, F1 = $0.8670$.
- **Operational Reality**: In real-world financial risk management, the decision boundary is tuned based on cost asymmetry rather than pure mathematical F1 optimization:
  - *Lowering the threshold* increases Recall (preventing high-dollar financial theft) at the expense of higher manual review queue volume.
  - *Raising the threshold* reduces customer friction (false declines) but tolerates higher fraud loss.

---

## 12. How to Run the Project

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Jupyter Notebook or VS Code with Jupyter extension

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd credit-card-fraud-detection
   ```
2. **Install Required Packages**:
   ```bash
   pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```
3. **Place the Dataset**:
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
   - Place `creditcard.csv` in the root project directory.
   *(Note: `creditcard.csv` is ~150 MB and should not be committed directly to GitHub).*
4. **Run the Notebook**:
   - Open and execute `Credit Card Fraud Detection - Classification.ipynb` sequentially from beginning to end.

---

## 13. Project Limitations & Future Improvements

- **Anonymized PCA Features**: Because features $V_1$ to $V_{28}$ are anonymized PCA projections, domain-specific features (such as merchant category codes, device fingerprints, transaction geographic distance, and user historical velocities) cannot be engineered.
- **Static Train-Test Partition**: The dataset reflects 48 hours of transactions. A time-based sequential validation scheme over extended months would better assess concept drift and evolving fraud patterns.
- **Hyperparameter Optimization**: Models were evaluated using fixed tree counts and standard configurations to ensure uniform benchmarking across imbalance strategies; Bayesian optimization or systematic grid search could further refine performance.
