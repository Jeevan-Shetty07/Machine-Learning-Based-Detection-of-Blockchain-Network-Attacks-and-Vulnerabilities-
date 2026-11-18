# Machine Learning-Based Detection of Blockchain Network Attacks and Vulnerabilities

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Reproducibility: Verified](https://img.shields.io/badge/Reproducibility-Verified-success.svg)](execution_logs.txt)
[![Models: RF | SVM | XGBoost](https://img.shields.io/badge/Models-RF%20|%20SVM%20|%20XGBoost-orange.svg)](#model-performance--comparative-results)

> **Reproducibility Confirmation:**  
> All source code, dataset artifacts, pre-computed model outputs, visual representations, and step-by-step execution logs are uploaded to this Git repository to confirm and guarantee the end-to-end reproducibility of the research work.

---

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Repository Structure](#-repository-structure)
3. [Dataset Description & Preprocessing](#-dataset-description--preprocessing)
4. [Machine Learning Models](#-machine-learning-models)
5. [Model Performance & Comparative Results](#-model-performance--comparative-results)
6. [Risk Scoring & Vulnerability Assessment](#-risk-scoring--vulnerability-assessment)
7. [Feature Importance & Behavioral Insights](#-feature-importance--behavioral-insights)
8. [Reproducibility & Execution Guide](#-reproducibility--execution-guide)
9. [Citation & References](#-citation--references)

---

## 📖 Overview

Blockchain systems face an expanding attack surface from illicit transactions, smart contract exploits, account takeovers, and fraudulent token transfers. This project establishes an end-to-end machine learning framework for detecting malicious blockchain network activity and identifying high-risk account addresses.

### Key Objectives:
- Extract and analyze behavioral transaction patterns on Ethereum/ERC-20 networks.
- Address class imbalance, missing telemetry, and noisy address identifiers.
- Train, evaluate, and benchmark multiple state-of-the-art classifiers (**Random Forest**, **Support Vector Machine**, and **XGBoost**).
- Establish a quantitative **Risk Scoring Framework (0 - 100)** to stratify blockchain addresses into Low, Medium, and High-risk categories.
- Provide comprehensive **execution logs** and a fully reproducible codebase.

---

## 📂 Repository Structure

```text
.
├── README.md                                    # Comprehensive project documentation & reproduction guide
├── blockchain_attack_dataset.csv                # Dataset with transaction features and attack labels
├── blockchain_network_attack_detection.ipynb    # Clean Jupyter Notebook with all cells, code, and outputs
├── untitled (2).ipynb                           # Original Colab execution notebook with full logs & figures
├── blockchain_attack_detection.py               # Standalone, end-to-end Python pipeline script
├── execution_logs.txt                           # Full text-based execution logs verifying all cell runs
├── top10_feature_importance.csv                 # Ranked feature contribution table from XGBoost
└── requirements.txt                             # Python dependencies and version specifications
```

---

## 📊 Dataset Description & Preprocessing

The primary dataset (`blockchain_attack_dataset.csv`) contains behavioral metrics extracted from Ethereum blockchain nodes:

- **Total Samples:** 9,841 transactions / addresses
- **Initial Features:** 51 attributes (reduced to 48 after cleaning)
- **Target Label (`FLAG`):**
  - `0`: Legitimate / Normal transactions (~77.8%)
  - `1`: Malicious / Attack / Vulnerability exploits (~22.2%)

### Data Pipeline:
1. **Duplicate Elimination:** Redundant transaction rows are identified and pruned.
2. **Identifier Removal:** Non-predictive identification columns (`Unnamed: 0`, `Index`, `Address`) are dropped to avoid data leakage and spurious memorization.
3. **Missing Value Imputation:**
   - Numerical attributes filled with `0`.
   - Categorical attributes (`ERC20 most sent token type`, `ERC20_most_rec_token_type`) imputed with `'Unknown'`.
4. **Categorical Encoding:** Token type attributes encoded via `LabelEncoder`.
5. **Data Splitting & Scaling:**
   - 80/20 stratified train/test split.
   - Standardized via `StandardScaler` ($\mu = 0, \sigma = 1$).

---

## 🤖 Machine Learning Models

Three distinct model families were trained and tested:
1. **Random Forest Classifier (Ensemble Bagging):**
   - Non-linear decision surfaces, robust to outliers and skewed distributions.
2. **Support Vector Machine (SVM - RBF Kernel):**
   - Maximizes the geometric margin between normal and attack behaviors in high-dimensional kernel space.
3. **XGBoost Classifier (Extreme Gradient Boosting):**
   - High-performance gradient-boosted decision trees optimizing binary logistic loss with regularization.

---

## 📈 Model Performance & Comparative Results

All metrics below reflect the test partition ($N = 1,969$) as recorded in [`execution_logs.txt`](execution_logs.txt):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 99.09% (`0.990858`) | **100.00%** (`1.000000`) | 95.87% (`0.958716`) | 97.89% (`0.978923`) |
| **Support Vector Machine (SVM)** | 95.58% (`0.955815`) | 89.93% (`0.899314`) | 90.14% (`0.901376`) | 90.03% (`0.900344`) |
| **XGBoost Classifier** | **99.19%** (`0.991874`) | **99.30%** (`0.992958`) | **97.02%** (`0.970183`) | **98.14%** (`0.981439`) |

### Key Findings:
- **XGBoost achieved the highest overall detection accuracy (99.19%) and F1-Score (98.14%)**, minimizing false negatives (Recall = 97.02%).
- **Random Forest achieved 100% Precision**, indicating that every address classified as an attack was indeed malicious (zero false positives).

---

## 🛡️ Risk Scoring & Vulnerability Assessment

Using the predicted probabilities from the optimal XGBoost model ($P(\text{Attack} = 1)$), a **Risk Score** is computed:

$$\text{Risk Score} = P(\text{Fraud}) \times 100$$

The addresses are categorized into risk tiers:
- **Low Risk ($0 \le \text{Score} \le 30$):** 1,537 addresses (78.06%)
- **Medium Risk ($30 < \text{Score} \le 70$):** 10 addresses (0.51%)
- **High Risk ($70 < \text{Score} \le 100$):** 422 addresses (21.43%)

### Risk Score Distribution:
- **Mean:** 21.76
- **Median (50th percentile):** 0.0066
- **Standard Deviation:** 40.92
- **Maximum:** 99.999

---

## 🔍 Feature Importance & Behavioral Insights

Analysis of gradient boosting feature attribution revealed the primary blockchain network attributes indicating attack patterns:

| Rank | Feature | Importance | Contribution (%) |
| :---: | :--- | :---: | :---: |
| 1 | `ERC20 most sent token type` | 0.316932 | **34.90%** |
| 2 | `ERC20_most_rec_token_type` | 0.268753 | **29.59%** |
| 3 | `Total ERC20 tnxs` | 0.186467 | **20.53%** |
| 4 | `ERC20 total ether sent` | 0.035860 | **3.95%** |
| 5 | `Time Diff between first and last (Mins)` | 0.029083 | **3.20%** |
| 6 | `ERC20 avg val sent` | 0.016623 | **1.83%** |
| 7 | `ERC20 uniq sent token name` | 0.016157 | **1.78%** |
| 8 | `Received Tnx` | 0.013736 | **1.51%** |
| 9 | `total transactions (including contract creation)` | 0.012552 | **1.38%** |
| 10 | `Unique Received From Addresses` | 0.012046 | **1.33%** |

> **Insight:** Over **85% of predictive power** stems from ERC-20 token transfer patterns, showing that modern blockchain attacks heavily involve token contract manipulations, rapid drainage, and novel token creation rather than plain ether transfers alone.

---

## 🚀 Reproducibility & Execution Guide

### Option 1: Run via Jupyter Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/Jeevan-Shetty07/Machine-Learning-Based-Detection-of-Blockchain-Network-Attacks-and-Vulnerabilities-.git
   cd Machine-Learning-Based-Detection-of-Blockchain-Network-Attacks-and-Vulnerabilities-
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook blockchain_network_attack_detection.ipynb
   ```

### Option 2: Run End-to-End CLI Pipeline
```bash
python blockchain_attack_detection.py
```

### Option 3: Inspect Frozen Execution Logs
To inspect verified execution outputs without rerunning heavy training cycles, review:
- [`execution_logs.txt`](execution_logs.txt): Plain-text log containing per-cell inputs, outputs, classification reports, and metrics.
- [`blockchain_network_attack_detection.ipynb`](blockchain_network_attack_detection.ipynb): Pre-rendered notebook with interactive charts and confusion matrices.

---

## 📜 Citation & References

```bibtex
@misc{blockchain_attack_detection_2026,
  author = {Jeevan Shetty},
  title = {Machine Learning-Based Detection of Blockchain Network Attacks and Vulnerabilities},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Jeevan-Shetty07/Machine-Learning-Based-Detection-of-Blockchain-Network-Attacks-and-Vulnerabilities-}}
}
```
