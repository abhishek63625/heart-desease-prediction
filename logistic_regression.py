import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score)

# Load the dataset
# Ensure the dataset file is in the same directory
try:
    df = pd.read_csv('Heart_Disease_Prediction.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Heart_Disease_Prediction.csv not found.")
    exit()

# ---------------------------------------------------------
# Step 1: Data Preparation
# ---------------------------------------------------------
print("\n" + "="*60)
print("LOGISTIC REGRESSION - HEART DISEASE CLASSIFICATION")
print("="*60)

print("\n--- Original Dataset Info ---")
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Create a copy for processing
df_processed = df.copy()

# ---------------------------------------------------------
# Step 2: Handle Categorical Variables
# ---------------------------------------------------------
print("\n--- Data Types ---")
print(df_processed.dtypes)

# Encode the target variable (Heart Disease: Presence=1, Absence=0)
print("\n--- Encoding Target Variable ---")
df_processed['Heart Disease'] = df_processed['Heart Disease'].map({'Presence': 1, 'Absence': 0})
print(f"Target variable encoded: Presence=1, Absence=0")
print(f"Value counts:")
print(df_processed['Heart Disease'].value_counts())
print(f"\nClass Distribution:")
print(f"  Class 0 (Absence): {(df_processed['Heart Disease']==0).sum()} ({(df_processed['Heart Disease']==0).sum()/len(df_processed)*100:.1f}%)")
print(f"  Class 1 (Presence): {(df_processed['Heart Disease']==1).sum()} ({(df_processed['Heart Disease']==1).sum()/len(df_processed)*100:.1f}%)")

# ---------------------------------------------------------
# Step 3: Prepare Features (X) and Target (y)
# ---------------------------------------------------------
print("\n--- Preparing Features and Target ---")

# Select all features except the target variable
X = df_processed.drop('Heart Disease', axis=1)
y = df_processed['Heart Disease']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeatures used in the model:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# ---------------------------------------------------------
# Step 4: Split Data into Training and Testing Sets
# ---------------------------------------------------------
print("\n--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_processed)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(df_processed)*100:.1f}%)")

# ---------------------------------------------------------
# Step 5: Feature Scaling (Important for Logistic Regression)
# ---------------------------------------------------------
print("\n--- Feature Scaling ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler")
print(f"Mean of scaled features (should be ~0): {X_train_scaled.mean():.6f}")
print(f"Std Dev of scaled features (should be ~1): {X_train_scaled.std():.6f}")

# ---------------------------------------------------------
# Step 6: Train Logistic Regression Model
# ---------------------------------------------------------
print("\n--- Training Logistic Regression Model ---")
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
print("Model training completed!")

# ---------------------------------------------------------
# Step 7: Make Predictions
# ---------------------------------------------------------
print("\n--- Making Predictions ---")
y_train_pred = log_model.predict(X_train_scaled)
y_test_pred = log_model.predict(X_test_scaled)

# Get prediction probabilities
y_train_proba = log_model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = log_model.predict_proba(X_test_scaled)[:, 1]

print(f"Predictions made on training set: {len(y_train_pred)} predictions")
print(f"Predictions made on testing set: {len(y_test_pred)} predictions")

# ---------------------------------------------------------
# Step 8: Model Evaluation
# ---------------------------------------------------------
print("\n" + "-"*60)
print("MODEL EVALUATION METRICS")
print("-"*60)

# Training Set Metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

# Testing Set Metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\n📊 TRAINING SET PERFORMANCE:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print(f"  AUC-ROC:   {train_auc:.4f}")

print("\n📊 TESTING SET PERFORMANCE:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  AUC-ROC:   {test_auc:.4f}")

# ---------------------------------------------------------
# Step 9: Confusion Matrix
# ---------------------------------------------------------
print("\n" + "-"*60)
print("CONFUSION MATRIX")
print("-"*60)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("\nTraining Set Confusion Matrix:")
print(cm_train)
print(f"  True Negatives (TN):  {cm_train[0, 0]}")
print(f"  False Positives (FP): {cm_train[0, 1]}")
print(f"  False Negatives (FN): {cm_train[1, 0]}")
print(f"  True Positives (TP):  {cm_train[1, 1]}")

print("\nTesting Set Confusion Matrix:")
print(cm_test)
print(f"  True Negatives (TN):  {cm_test[0, 0]}")
print(f"  False Positives (FP): {cm_test[0, 1]}")
print(f"  False Negatives (FN): {cm_test[1, 0]}")
print(f"  True Positives (TP):  {cm_test[1, 1]}")

# ---------------------------------------------------------
# Step 10: Classification Report
# ---------------------------------------------------------
print("\n" + "-"*60)
print("CLASSIFICATION REPORT (TESTING SET)")
print("-"*60)
print(classification_report(y_test, y_test_pred, target_names=['Absence', 'Presence']))

# ---------------------------------------------------------
# Step 11: Model Coefficients
# ---------------------------------------------------------
print("\n" + "-"*60)
print("MODEL COEFFICIENTS")
print("-"*60)
print(f"Intercept: {log_model.intercept_[0]:.4f}")
print("\nFeature Coefficients:")
coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
})
coefficients_df = coefficients_df.sort_values('Coefficient', ascending=False)
print(coefficients_df.to_string(index=False))

print("\nInterpretation of Coefficients:")
print("- Positive coefficient: Feature increases heart disease probability")
print("- Negative coefficient: Feature decreases heart disease probability")
print("- Larger absolute value: Feature has stronger influence on outcome")

# ---------------------------------------------------------
# Step 12: Visualizations
# ---------------------------------------------------------
print("\n--- Generating Visualizations ---")

# Visualization 1: Confusion Matrix Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training Confusion Matrix
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Training Set: Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['Absence', 'Presence'])
axes[0].set_yticklabels(['Absence', 'Presence'])

# Testing Confusion Matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Testing Set: Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['Absence', 'Presence'])
axes[1].set_yticklabels(['Absence', 'Presence'])

plt.tight_layout()
plt.show()

# Visualization 2: ROC Curves
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, label=f'Training AUC = {train_auc:.4f}', linewidth=2)
plt.plot(fpr_test, tpr_test, label=f'Testing AUC = {test_auc:.4f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualization 3: Feature Importance (Coefficients)
plt.figure(figsize=(10, 6))
top_features = coefficients_df.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Top 10 Features by Coefficient Magnitude')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Visualization 4: Performance Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
train_scores = [train_accuracy, train_precision, train_recall, train_f1, train_auc]
test_scores = [test_accuracy, test_precision, test_recall, test_f1, test_auc]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, train_scores, width, label='Training', alpha=0.8)
plt.bar(x + width/2, test_scores, width, label='Testing', alpha=0.8)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Metrics Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 1.1])
for i, v in enumerate(train_scores):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(test_scores):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# Visualization 5: Prediction Probability Distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_train_proba[y_train == 0], bins=20, alpha=0.6, label='Absence', color='blue')
plt.hist(y_train_proba[y_train == 1], bins=20, alpha=0.6, label='Presence', color='red')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Training Set: Prediction Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_test_proba[y_test == 0], bins=20, alpha=0.6, label='Absence', color='blue')
plt.hist(y_test_proba[y_test == 1], bins=20, alpha=0.6, label='Presence', color='red')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Testing Set: Prediction Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 13: Model Summary
# ---------------------------------------------------------
print("\n" + "="*60)
print("LOGISTIC REGRESSION - MODEL SUMMARY")
print("="*60)

print(f"""
✓ Model Type: Logistic Regression (Classification)
✓ Number of Features: {X.shape[1]}
✓ Training Samples: {X_train.shape[0]}
✓ Testing Samples: {X_test.shape[0]}
✓ Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)
✓ Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
✓ Training AUC-ROC: {train_auc:.4f}
✓ Testing AUC-ROC: {test_auc:.4f}

Model Performance Interpretation:
- Accuracy: {test_accuracy*100:.2f}% of predictions are correct
- Precision: {test_precision*100:.2f}% of positive predictions are correct
- Recall: {test_recall*100:.2f}% of actual positives are identified
- F1-Score: {test_f1:.4f} (harmonic mean of precision and recall)
- AUC-ROC: {test_auc:.4f} (ability to distinguish between classes)

Key Findings:
""")

# Top positive and negative features
top_positive = coefficients_df[coefficients_df['Coefficient'] > 0].head(3)
top_negative = coefficients_df[coefficients_df['Coefficient'] < 0].head(3)

print("\nTop 3 Features INCREASING Heart Disease Risk:")
for idx, (_, row) in enumerate(top_positive.iterrows(), 1):
    print(f"  {idx}. {row['Feature']}: {row['Coefficient']:.4f}")

print("\nTop 3 Features DECREASING Heart Disease Risk:")
for idx, (_, row) in enumerate(top_negative.iterrows(), 1):
    print(f"  {idx}. {row['Feature']}: {row['Coefficient']:.4f}")

print("\nModel Recommendations:")
print(f"- Model is {'suitable' if test_accuracy > 0.75 else 'needs improvement'} for clinical use")
print(f"- Sensitivity (Recall): {test_recall*100:.2f}% - identifies actual disease cases")
print(f"- Specificity: {(cm_test[0, 0]/(cm_test[0, 0]+cm_test[0, 1])*100):.2f}% - identifies healthy cases")

print("\n" + "="*60)
print("Logistic Regression analysis completed successfully!")
print("="*60)
