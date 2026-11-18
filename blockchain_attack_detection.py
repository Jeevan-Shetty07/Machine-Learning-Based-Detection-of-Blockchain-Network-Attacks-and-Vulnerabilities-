"""
Machine Learning-Based Detection of Blockchain Network Attacks and Vulnerabilities
Reproducible Pipeline Script
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def main():
    print("=" * 70)
    print("Machine Learning-Based Detection of Blockchain Network Attacks")
    print("Reproducible Pipeline Execution")
    print("=" * 70)
    
    # 1. Load Dataset
    dataset_path = 'blockchain_attack_dataset.csv'
    print(f"\n[1] Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    print(f"    Initial Dataset Shape: {df.shape}")
    
    # 2. Data Cleaning
    print("\n[2] Preprocessing and Cleaning...")
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"    Duplicates removed: {initial_len - len(df)}")
    
    # Drop identifiers
    id_cols = [c for c in ['Unnamed: 0', 'Index', 'Address'] if c in df.columns]
    if id_cols:
        df.drop(id_cols, axis=1, inplace=True)
        print(f"    Dropped metadata/identifier columns: {id_cols}")
    
    print(f"    Cleaned Dataset Shape: {df.shape}")
    print(f"    Target ('FLAG') distribution:\n{df['FLAG'].value_counts().to_dict()}")
    
    # Handle missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    print(f"    Missing values handled: Remaining nulls = {df.isnull().sum().sum()}")
    
    # Categorical Encoding
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"    Encoded categorical columns: {list(cat_cols)}")
    
    # 3. Train/Test Split
    print("\n[3] Splitting features and target...")
    X = df.drop('FLAG', axis=1)
    y = df['FLAG']
    print(f"    Features shape: {X.shape}, Target shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model Training & Evaluation
    print("\n[4] Training Models...")
    
    # Model 1: Random Forest
    print("\n--> Training Random Forest Classifier...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    
    print(f"    Random Forest - Acc: {rf_acc:.6f}, Prec: {rf_prec:.6f}, Rec: {rf_rec:.6f}, F1: {rf_f1:.6f}")
    
    # Model 2: SVM
    print("\n--> Training Support Vector Machine (SVM)...")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred)
    svm_rec = recall_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred)
    
    print(f"    SVM           - Acc: {svm_acc:.6f}, Prec: {svm_prec:.6f}, Rec: {svm_rec:.6f}, F1: {svm_f1:.6f}")
    
    # Model 3: XGBoost
    print("\n--> Training XGBoost Classifier...")
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_prec = precision_score(y_test, xgb_pred)
    xgb_rec = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    
    print(f"    XGBoost       - Acc: {xgb_acc:.6f}, Prec: {xgb_prec:.6f}, Rec: {xgb_rec:.6f}, F1: {xgb_f1:.6f}")
    
    # Summary Table
    results = pd.DataFrame({
        'Model': ['Random Forest', 'SVM', 'XGBoost'],
        'Accuracy': [rf_acc, svm_acc, xgb_acc],
        'Precision': [rf_prec, svm_prec, xgb_prec],
        'Recall': [rf_rec, svm_rec, xgb_rec],
        'F1 Score': [rf_f1, svm_f1, xgb_f1]
    })
    
    print("\n[5] Benchmark Performance Comparison:")
    print(results.to_string(index=False))
    
    # 5. Risk Scoring & Probability Analysis
    print("\n[6] Risk Scoring Calculation...")
    fraud_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    risk_score = fraud_proba * 100
    
    risk_df = pd.DataFrame({
        'Actual_Class': y_test.values,
        'Fraud_Probability': fraud_proba,
        'Risk_Score': risk_score
    })
    
    risk_df['Risk_Category'] = pd.cut(
        risk_df['Risk_Score'],
        bins=[-1, 30, 70, 100],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    print("    Risk Category Counts:")
    print(risk_df['Risk_Category'].value_counts())
    
    # 6. Feature Importance
    print("\n[7] Extracting Feature Importance from XGBoost...")
    feat_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    top10 = feat_imp.head(10).copy()
    top10['Contribution (%)'] = (top10['Importance'] / top10['Importance'].sum()) * 100
    
    print("    Top 10 Important Features:")
    print(top10.to_string(index=False))
    
    top10.to_csv('top10_feature_importance.csv', index=False)
    print("\n[+] Saved top10_feature_importance.csv")
    print("=" * 70)
    print("Execution and reproducibility check completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
