
# Credit Card Fraud Detection: Classification & Imbalance Benchmarking

An end-to-end machine learning project comparing classification models and class-imbalance techniques for detecting fraudulent credit card transactions. The project focuses on model evaluation, precision-recall trade-offs, and decision-threshold selection.

---

## 1. Project Overview & Problem Statement

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a tiny fraction of all transactions. A useful model must identify fraud while minimizing false alarms on legitimate transactions.

This project benchmarks three machine learning classifiers across three class-imbalance strategies, evaluates their performance on validation data, and performs a final evaluation using a separate held-out test set.

## 2. Dataset Overview

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Total transactions**: 284,807.
- **Legitimate transactions (Class 0)**: 284,315 (99.828%).
- **Fraudulent transactions (Class 1)**: 492 (0.172%).
- **Imbalance ratio**: Approximately 578 legitimate transactions per fraudulent transaction.

### Features

- `V1` to `V28`: Anonymized numerical features obtained through PCA.
- `Time`: Elapsed seconds between a transaction and the first transaction in the dataset.
- `Amount`: Transaction amount as recorded in the dataset; the documentation does not explicitly specify the currency.
- `Class`: Target variable, where 1 indicates fraud and 0 indicates a legitimate transaction.

---

## 3. Key Challenge: Extreme Class Imbalance

Fraudulent transactions account for only 0.172% of the dataset, making accuracy alone an unreliable measure of model performance.

- **Misleading accuracy**: A classifier predicting every transaction as legitimate would achieve approximately 99.828% accuracy while detecting no fraud.
- **Precision and recall trade-off**: Increasing the number of detected fraud cases can also increase false positives.
- **PR-AUC**: Precision-Recall AUC is especially useful for assessing performance on the minority fraud class.
- **ROC-AUC**: Provides a complementary measure of the model's ability to distinguish between legitimate and fraudulent transactions.

---

## 4. Technologies Used

- **Language**: Python
- **Data manipulation**: NumPy, Pandas
- **Machine learning**: Scikit-learn, XGBoost
- **Imbalanced data handling**: Imbalanced-learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook / VS Code

---

## 5. Exploratory Data Analysis (EDA)

The exploratory analysis examines class distribution, transaction amounts, and feature relationships.

1. **Class distribution**:
   - Analyzed the distribution of legitimate and fraudulent transactions.
   - Identified the approximately 578:1 class imbalance.

2. **Transaction amount comparison**:
   - **Legitimate**: Mean = 88.29, median = 22.00, 75th percentile = 77.05, maximum = 25,691.16.
   - **Fraudulent**: Mean = 122.21, median = 9.25, 75th percentile = 105.89, maximum = 2,125.87.
   - Fraudulent transactions have a smaller median transaction amount than legitimate transactions in this dataset.

3. **Temporal feature handling**:
   - `Time` represents elapsed seconds since the first transaction, not actual clock time.
   - The derived `Hour` feature was excluded because it does not reliably represent the actual hour of day.
   - No conclusions about diurnal fraud patterns are drawn from these features.

4. **Feature relationships**:
   - Analyzed relationships between numerical features and the fraud target.
   - Observed class separation in features including `V14`, `V17`, `V12`, and `V10`.

---

## 6. Feature Engineering & Preprocessing

### Feature Selection

- Used `V1` to `V28` and `Amount` as model features, for a total of 29 features.
- Excluded `Time` and the derived `Hour` feature from model training.

### Stratified Train-Validation-Test Split

The dataset was divided into three partitions using stratified sampling and `random_state=42`.

| Partition | Transactions | Fraudulent transactions |
|---|---:|---:|
| Training | 170,883 | 295 |
| Validation | 56,962 | 99 |
| Testing | 56,962 | 98 |

### Preprocessing and Leakage Prevention

- Fitted `RobustScaler` exclusively on the training data.
- Applied the fitted scaler to the validation and test sets without refitting.
- Compared model configurations using validation-set metrics.
- Selected the model using validation PR-AUC and selected the decision threshold using validation F1-score.
- Reserved the test set for the final evaluation after model and threshold selection.

---

## 7. Model Architectures & Imbalance Handling Strategies

### Model Architectures

- **Logistic Regression**: Linear classification model used as a baseline.
- **Random Forest**: Ensemble of 100 decision trees that captures nonlinear relationships.
- **XGBoost**: Gradient-boosted decision trees using 100 estimators.

### Imbalance Handling Paradigms

Three approaches were evaluated to address the extreme class imbalance.

1. **Baseline**
   - Models trained on the original, unweighted training data.
   - No resampling or class-specific weighting was applied.

2. **Cost-Sensitive / Class-Weighted**
   - Logistic Regression: `class_weight='balanced'`
   - Random Forest: `class_weight='balanced'`
   - XGBoost: `scale_pos_weight = 578.26`, calculated from the training set's negative-to-positive class ratio.

3. **SMOTE (Synthetic Minority Oversampling)**
   - Applied exclusively to the training data.
   - Increased the training set from 170,883 transactions to 341,176 resampled transactions.
   - The resulting resampled training data contained 170,588 samples in each class.
   - Validation and test sets were left untouched by SMOTE.

---

## 8. Evaluation Metrics

The held-out test set contains 56,864 legitimate transactions and 98 fraudulent transactions.

The project uses the following evaluation metrics:

- **Precision**: Proportion of transactions flagged as fraud that were actually fraudulent.
- **Recall**: Proportion of actual fraudulent transactions identified by the model.
- **F1-score**: Harmonic mean of precision and recall.
- **PR-AUC (Average Precision)**: Summarizes precision-recall performance across different classification thresholds.
- **ROC-AUC**: Measures the model's ability to distinguish between the two classes across classification thresholds.

PR-AUC is emphasized because of the dataset's extreme class imbalance.

---

## 9. Experimental Benchmark Results

Nine model configurations were evaluated using three classifiers and three imbalance-handling strategies.

### Model Selection

- **Selected model**: XGBoost with SMOTE.
- **Selection criterion**: Highest validation PR-AUC among the nine evaluated configurations.
- **Validation PR-AUC**: 0.8087.
- **Selected decision threshold**: 0.9906, chosen using validation data to maximize F1-score.

### Final Held-Out Test Results

The selected model was evaluated on the independent test set using the validation-selected threshold.

| Metric | Final test result |
|---|---:|
| Precision | 0.9259 |
| Recall | 0.7653 |
| F1-score | 0.8380 |
| PR-AUC | 0.8623 |
| ROC-AUC | 0.9814 |

Precision, recall, and F1-score reflect predictions at the selected threshold. PR-AUC and ROC-AUC evaluate discrimination across thresholds.

### Selected Validation Comparisons

The following results illustrate differences between class-weighted and SMOTE-based approaches.

| Model | Strategy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | SMOTE | 0.7600 | 0.7677 | 0.7638 | 0.8087 | 0.9675 |
| XGBoost | Class-weighted | 0.9167 | 0.7778 | 0.8415 | 0.8069 | 0.9671 |
| Random Forest | SMOTE | 0.8652 | 0.7778 | 0.8191 | 0.7975 | 0.9567 |
| Random Forest | Class-weighted | 0.8916 | 0.7475 | 0.8132 | 0.7951 | 0.9463 |
| Logistic Regression | Class-weighted | 0.0571 | 0.8990 | 0.1074 | 0.6826 | 0.9758 |
| Logistic Regression | SMOTE | 0.0589 | 0.8788 | 0.1104 | 0.6751 | 0.9724 |

The table presents the six class-weighted and SMOTE configurations for which validation results are summarized here. The three baseline configurations were also evaluated; their results can be inspected in the notebook.

---

## 10. Key Empirical Findings

1. **Model selection**:
   - XGBoost with SMOTE achieved the highest validation PR-AUC among the nine configurations, at 0.8087.
   - XGBoost with class weighting achieved a similar validation PR-AUC of 0.8069.
   - Model selection was based on validation PR-AUC, followed by threshold selection using validation F1-score.

2. **Class weighting and SMOTE trade-offs**:
   - Class-weighted Logistic Regression achieved validation recall of 0.8990 but precision of only 0.0571.
   - SMOTE-based Logistic Regression achieved validation recall of 0.8788 but precision of only 0.0589.
   - These results show that high recall alone does not guarantee an effective fraud-detection system because false positives also matter.

3. **Final test performance**:
   - At the selected threshold of 0.9906, XGBoost with SMOTE achieved test precision of 0.9259, recall of 0.7653, and F1-score of 0.8380.
   - The model correctly identified 75 of the 98 fraudulent transactions in the test set, while incorrectly flagging 6 legitimate transactions.

4. **Evaluation limitations**:
   - Results are based on one stratified random train-validation-test split.
   - The dataset covers a limited period and may not represent evolving fraud patterns.
   - Threshold selection based on F1-score does not necessarily minimize real-world financial losses or operational costs.

---

## 11. Confusion Matrix & Decision-Threshold Analysis

### Final Test Confusion Matrix

The selected XGBoost + SMOTE model was evaluated at the threshold of `0.9906` selected using validation data.

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 56,858 | 6 |
| Actual fraud | 23 | 75 |

### Operational Breakdown

- **True Negatives**: 56,858 legitimate transactions correctly identified.
- **False Positives**: 6 legitimate transactions incorrectly flagged as fraud.
- **False Negatives**: 23 fraudulent transactions missed by the model.
- **True Positives**: 75 fraudulent transactions correctly identified.

### Threshold Selection

- Candidate thresholds were evaluated using validation-set probabilities.
- The threshold of `0.9906` was selected to maximize validation F1-score.
- The held-out test set was used for final evaluation after model and threshold selection.

### Practical Trade-Off

- Lowering the decision threshold generally increases recall but may also increase false alarms.
- Raising the threshold may improve precision while allowing more fraud to go undetected.
- A production fraud-detection system should consider fraud losses, investigation costs, and customer friction when choosing an operational threshold.

---

## 12. How to Run the Project

### Prerequisites

- Python 3.10 or newer
- Jupyter Notebook or VS Code with the Jupyter extension

### Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Ritvik440/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install the required packages**:

   ```bash
   pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```

3. **Download the dataset**:
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
   - Place the CSV file in the notebook's working directory.
   - The dataset is approximately 150 MB and should not be committed directly to GitHub.

4. **Run the notebook**:
   - Open `Credit Card Fraud Detection - Revised.ipynb`.
   - Execute the notebook cells sequentially from beginning to end.

---

## 13. Project Limitations & Future Improvements

- **Anonymized PCA features**: The anonymized features `V1` to `V28` limit domain-specific feature engineering.
- **Static random train-validation-test split**: The dataset covers approximately 48 hours of transactions. A time-based evaluation using longer-period transaction data would better assess changing fraud patterns.
- **Hyperparameter optimization**: The current experiments use fixed model configurations. Grid search or Bayesian optimization could help explore better-performing configurations.
- **Operational threshold selection**: Future work could incorporate explicit costs for missed fraud, false alarms, and manual investigation to choose a threshold based on business requirements.
