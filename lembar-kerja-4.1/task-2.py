# ==========================================
# Loan Prediction Classification (Scikit-learn)
# Dataset: Kaggle Loan Prediction (train & test)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")

# 2. Gabungkan untuk preprocessing yang konsisten
train_df['source'] = 'train'
test_df['source'] = 'test'
test_df['Loan_Status'] = np.nan  # Tambahkan kolom target agar konsisten
combined = pd.concat([train_df, test_df], ignore_index=True)

# 3. Tangani Missing Values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']:
    combined[col] = combined[col].fillna(combined[col].mode()[0])
combined['LoanAmount'] = combined['LoanAmount'].fillna(combined['LoanAmount'].median())
combined['Loan_Status'] = combined['Loan_Status'].fillna('N')

# 4. Label Encoding untuk kolom kategorikal
label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']
le = LabelEncoder()
for col in label_cols:
    combined[col] = le.fit_transform(combined[col])

# 5. Split kembali ke train/test setelah preprocessing
train_cleaned = combined[combined['source'] == 'train'].drop(['source', 'Loan_ID'], axis=1)
test_cleaned = combined[combined['source'] == 'test'].drop(['source', 'Loan_ID', 'Loan_Status'], axis=1)

# 6. Pisahkan fitur dan target
X = train_cleaned.drop('Loan_Status', axis=1)
y = train_cleaned['Loan_Status']

# 7. Split data latih dan uji untuk evaluasi
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# 8. Normalisasi fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# 9. Latih Model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 10. Evaluasi Model
y_pred = model.predict(X_valid_scaled)
print("Akurasi:", accuracy_score(y_valid, y_pred))
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))

# 11. Visualisasi Confusion Matrix
cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['N', 'Y'], yticklabels=['N', 'Y'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()