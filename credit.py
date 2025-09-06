# Credit Card Fraud Detection Project with Visualization

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv("D:\creditcard.csv")

# 3. Explore Dataset
print("Shape of dataset:", df.shape)
print(df['Class'].value_counts())

# 4. Bar Graph: Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Transaction Class Distribution")
plt.xticks([0,1], ['Not Fraud (0)', 'Fraud (1)'])
plt.ylabel("Number of Transactions")
plt.show()

# 5. Feature Scaling (scale Amount and Time)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

# 6. Split Data
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE, counts of label '1':", sum(y_train_res==1))
print("After SMOTE, counts of label '0':", sum(y_train_res==0))

# Optional: Bar Graph after SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res)
plt.title("Transaction Class Distribution After SMOTE")
plt.xticks([0,1], ['Not Fraud (0)', 'Fraud (1)'])
plt.ylabel("Number of Transactions")
plt.show()

# 8. Train Models

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_res, y_train_res)

# 9. Evaluate Models
models = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print(f"--- {name} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
    print("\n")

# 10. Feature Importance for Random Forest
importances = rf.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Feature Importances - Random Forest")
plt.show()
