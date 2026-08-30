# Credit Card Fraud Detection Using CatBoost

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small proportion of total transactions.

This project develops a machine learning pipeline to identify potentially fraudulent transactions using the **IEEE-CIS Fraud Detection dataset**. The project focuses on exploratory analysis, handling class imbalance, preserving the temporal nature of transactions, threshold optimization, and model interpretability using SHAP.

The final model uses **CatBoost**, a gradient boosting algorithm that can effectively handle datasets containing both numerical and categorical features.

---

## 🎯 Objectives

* Identify fraudulent transactions from highly imbalanced transaction data.
* Combine transaction-level and identity-related information.
* Analyze patterns associated with fraudulent activity.
* Build a CatBoost classification model with class imbalance handling.
* Optimize the classification threshold using validation data.
* Evaluate the final model on a temporally held-out test set.
* Interpret model predictions using SHAP.

---

## 📊 Dataset

The project uses the **IEEE-CIS Fraud Detection dataset**.

Two datasets are used:

* `train_transaction.csv` — transaction-level information.
* `train_identity.csv` — identity and device-related information.

The datasets are joined using `TransactionID` with a **left join**, preserving all transactions even when corresponding identity information is unavailable.

The merged dataset contains **590,540 transactions** and **434 features** before additional engineered features.

The original datasets are not included in this repository.

---

## 🔎 Exploratory Data Analysis

The analysis examined several dimensions of fraudulent behavior:

* Missing-value patterns
* Transaction amount distribution
* Transaction amount distribution by fraud status
* Fraud rate by card type
* Fraud rate by product
* Fraud rate by device type
* Fraud rate by email domain
* Fraud rate based on identity-data availability
* Fraud rate by transaction hour
* Transaction amount buckets
* Numerical feature correlation with the fraud target

### Identity Data Availability

An `has_identity` feature was created to indicate whether identity information was available for a transaction.

The analysis showed a substantially higher fraud rate among transactions with identity information available compared with transactions without identity information. This indicates that identity-data availability contains potentially useful information for fraud detection.

### Transaction Timing

`TransactionDT` was converted into time-based features including:

* `transaction_hour`
* `transaction_day`

Fraud rates were then analyzed across transaction hours to investigate whether fraudulent activity varies throughout the day.

---

## 🛠️ Feature Engineering & Preprocessing

### Temporal Features

The original `TransactionDT` variable was used to derive:

* `transaction_hour`
* `transaction_day`

These features allow the model to capture temporal patterns in transaction behavior.

### Identity Availability

A binary `has_identity` feature was created based on whether identity information was available.

### Transaction Amount Bucketing

`TransactionAmt` was divided into five frequency-based buckets using quantile binning to examine whether fraud rates vary across transaction-value ranges.

### Categorical Features

Categorical variables were converted to strings and missing values were explicitly represented so that they could be processed by CatBoost.

### Class Imbalance

Fraudulent transactions represent only a small proportion of the dataset.

Rather than using SMOTE, **class weighting** was applied during CatBoost training to assign greater importance to fraudulent transactions.

The fraud class weight was calculated from the ratio of legitimate to fraudulent transactions in the training set.

---

## ⏱️ Temporal Train / Validation / Test Split

To reduce the risk of temporal leakage, transactions were first sorted chronologically using `TransactionDT`.

The dataset was then divided into:

* **70% Training**
* **15% Validation**
* **15% Test**

The validation set was used for model evaluation and classification-threshold optimization.

The test set was kept completely separate until final evaluation.

This approach better represents a real-world scenario where historical transactions are used to train a model and future transactions are used for evaluation.

---

## 🤖 Model Development

### Baseline Model

A baseline CatBoost classifier was trained using:

* 300 iterations
* Learning rate: 0.05
* Depth: 6
* Class weighting
* AUC as the evaluation metric
* Early stopping

The baseline model was evaluated on the validation set using:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score
* Confusion matrix

The baseline model was also used to establish the initial threshold-selection workflow.

### Improved Model

An improved CatBoost classifier was then trained using:

* 600 iterations
* Learning rate: 0.05
* Depth: 6
* Class weighting
* AUC as the evaluation metric
* Early stopping

The improved model was subsequently evaluated using the temporally held-out test set.

---

## 🎚️ Classification Threshold Optimization

The default classification threshold of 0.5 is not necessarily optimal for an imbalanced fraud detection problem.

For the improved model, fraud probabilities were generated on the **validation set** and multiple classification thresholds were evaluated.

The threshold maximizing validation F1-score was selected:

**Best validation threshold: 0.7110**

Validation performance at this threshold:

| Metric    | Validation |
| --------- | ---------: |
| Precision |     55.80% |
| Recall    |     43.62% |
| F1-score  |     48.97% |

The selected threshold was then **fixed before evaluating the model on the unseen test set**.

---

## 📈 Final Test Performance

The final model was evaluated on the unseen, temporally held-out test set using the validation-selected threshold.

| Metric    |       Test |
| --------- | ---------: |
| ROC-AUC   | **0.8707** |
| PR-AUC    | **0.4476** |
| Precision | **50.92%** |
| Recall    | **43.14%** |
| F1-score  | **46.71%** |
| Accuracy  | **96.57%** |

### Confusion Matrix

|                       | Predicted Legitimate | Predicted Fraud |
| --------------------- | -------------------: | --------------: |
| **Actual Legitimate** |               84,216 |           1,282 |
| **Actual Fraud**      |                1,753 |           1,330 |

The model correctly identified **1,330 fraudulent transactions** while generating **1,282 false-positive alerts**.

It missed **1,753 fraudulent transactions**, highlighting the inherent trade-off between detecting more fraud and limiting false alerts.

### Business Interpretation

The final model achieves approximately **51% precision and 43% recall** at the selected operating threshold.

In practical terms, roughly half of the transactions flagged as fraudulent are actually fraudulent, while the model identifies around 43% of fraudulent transactions in the held-out test set.

For a real financial institution, the appropriate operating threshold would ultimately depend on the relative cost of:

* Missing a fraudulent transaction (**false negative**)
* Blocking or reviewing a legitimate transaction (**false positive**)

Therefore, future threshold optimization could use an explicit financial cost function rather than optimizing F1-score alone.

---

## 🔍 SHAP Model Interpretability

SHAP (**SHapley Additive exPlanations**) was used to understand the features that most strongly influence the CatBoost model's predictions.

The analysis was performed using a representative sample of **1,000 test transactions**.

The top features by mean absolute SHAP value were:

| Rank | Feature         | Mean Absolute SHAP |
| ---: | --------------- | -----------------: |
|    1 | D2              |             0.1853 |
|    2 | id_33           |             0.1817 |
|    3 | TransactionAmt  |             0.1630 |
|    4 | C13             |             0.1583 |
|    5 | card2           |             0.1534 |
|    6 | C14             |             0.1508 |
|    7 | C1              |             0.1502 |
|    8 | card6           |             0.1472 |
|    9 | card1           |             0.1283 |
|   10 | P_emaildomain   |             0.1233 |
|   11 | C5              |             0.1202 |
|   12 | transaction_day |             0.1160 |
|   13 | DeviceInfo      |             0.1094 |
|   14 | id_38           |             0.1005 |
|   15 | TransactionDT   |             0.1001 |
|   16 | C11             |             0.0981 |
|   17 | D15             |             0.0971 |
|   18 | M5              |             0.0935 |
|   19 | card5           |             0.0869 |
|   20 | C2              |             0.0858 |

### Interpretation

The SHAP analysis shows that the model relies on a combination of:

* Transaction characteristics such as `TransactionAmt`
* Card-related attributes such as `card1`, `card2`, `card5`, and `card6`
* Transaction-history/count features such as `C1`, `C2`, `C5`, `C11`, `C13`, and `C14`
* Identity and device-related information such as `id_33`, `DeviceInfo`, and `id_38`
* Temporal information such as `transaction_day` and `TransactionDT`

The prominence of these feature groups indicates that the model is capturing multiple dimensions of transaction behavior rather than relying on a single fraud signal.

The SHAP beeswarm plot additionally shows the direction of individual feature contributions: positive SHAP values push the model toward a higher fraud prediction, while negative SHAP values push it toward a lower fraud prediction.

> SHAP explains the model's behavior and does not establish that an individual feature causes fraud.

---

## 💼 Business Value

A fraud detection model can help financial institutions:

* Prioritize suspicious transactions for investigation.
* Reduce manual review workload.
* Identify transaction patterns associated with fraudulent activity.
* Support risk-based transaction monitoring.
* Balance fraud prevention with unnecessary customer friction.

The project demonstrates why fraud detection should be treated as a **decision and cost-optimization problem**, rather than simply maximizing classification accuracy.

---

## ⚠️ Limitations

* The dataset represents a specific fraud-detection environment and may not generalize directly to other financial institutions.
* The selected threshold was optimized using F1-score rather than an explicit monetary cost function.
* The model does not incorporate real-time transaction streams or production monitoring.
* Fraud patterns may evolve over time, causing model performance to deteriorate.
* Additional behavioral and velocity features could potentially improve fraud detection.
* The model's probabilities have not been specifically calibrated for downstream risk scoring.

---

## 🔮 Future Improvements

Potential improvements include:

* Cost-sensitive threshold optimization.
* Probability calibration.
* Transaction velocity and behavioral features.
* Time-window aggregation features.
* Model comparison with LightGBM or XGBoost.
* False-positive and false-negative investigation.
* Model drift monitoring.
* Real-time fraud scoring infrastructure.

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* CatBoost
* SHAP
* Jupyter Notebook

---

## 📁 Repository Structure

```text
fraud-detection/
│
├── fraud_detection.ipynb
├── requirements.txt
├── README.md
└── models/
    └── improved_model.cbm
```

---

## 👤 Author

**Srihari Sai Marepalli**

End-to-end machine learning project focused on fraud detection, imbalanced classification, temporal validation, threshold optimization, and model interpretability.
