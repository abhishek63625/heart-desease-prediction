import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
print("MULTIPLE LINEAR REGRESSION - HEART DISEASE PREDICTION")
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
print(f"Value counts:\n{df_processed['Heart Disease'].value_counts()}")

# ---------------------------------------------------------
# Step 3: Prepare Features (X) and Target (y)
# ---------------------------------------------------------
print("\n--- Preparing Features and Target ---")

# Select features for multiple linear regression
# We'll use all features except the target variable
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_processed)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(df_processed)*100:.1f}%)")

# ---------------------------------------------------------
# Step 5: Train Multiple Linear Regression Model
# ---------------------------------------------------------
print("\n--- Training Multiple Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed!")

# ---------------------------------------------------------
# Step 6: Make Predictions
# ---------------------------------------------------------
print("\n--- Making Predictions ---")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"Predictions made on training set: {len(y_train_pred)} predictions")
print(f"Predictions made on testing set: {len(y_test_pred)} predictions")

# ---------------------------------------------------------
# Step 7: Model Evaluation
# ---------------------------------------------------------
print("\n" + "-"*60)
print("MODEL EVALUATION METRICS")
print("-"*60)

# Training Set Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Testing Set Metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n📊 TRAINING SET PERFORMANCE:")
print(f"  Mean Squared Error (MSE):  {train_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"  R² Score: {train_r2:.4f} ({train_r2*100:.2f}%)")

print("\n📊 TESTING SET PERFORMANCE:")
print(f"  Mean Squared Error (MSE):  {test_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"  R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")

print("\n" + "-"*60)
print("MODEL COEFFICIENTS")
print("-"*60)
print(f"Intercept: {model.intercept_:.4f}")
print("\nFeature Coefficients:")
coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
coefficients_df = coefficients_df.sort_values('Coefficient', ascending=False)
print(coefficients_df.to_string(index=False))

print("\nInterpretation of Coefficients:")
print("- Positive coefficient: Feature increases heart disease probability")
print("- Negative coefficient: Feature decreases heart disease probability")
print("- Larger absolute value: Feature has stronger influence on outcome")

# ---------------------------------------------------------
# Step 8: Visualizations
# ---------------------------------------------------------
print("\n--- Generating Visualizations ---")

# Visualization 1: Actual vs Predicted Values (Training Set)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Training Set: Actual vs Predicted')
plt.grid(True, alpha=0.3)

# Visualization 2: Actual vs Predicted Values (Testing Set)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Testing Set: Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualization 3: Residuals Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
residuals_train = y_train - y_train_pred
plt.scatter(y_train_pred, residuals_train, alpha=0.6, color='blue')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Training Set: Residuals Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals_test = y_test - y_test_pred
plt.scatter(y_test_pred, residuals_test, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Testing Set: Residuals Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualization 4: Feature Importance (Coefficients)
plt.figure(figsize=(10, 6))
top_features = coefficients_df.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Top 10 Features by Coefficient Magnitude')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Visualization 5: Distribution of Residuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(residuals_train, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Training Set: Distribution of Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals_test, bins=20, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Testing Set: Distribution of Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 9: Model Summary
# ---------------------------------------------------------
print("\n" + "="*60)
print("MULTIPLE LINEAR REGRESSION - MODEL SUMMARY")
print("="*60)

print(f"""
✓ Model Type: Multiple Linear Regression
✓ Number of Features: {X.shape[1]}
✓ Training Samples: {X_train.shape[0]}
✓ Testing Samples: {X_test.shape[0]}
✓ Training R² Score: {train_r2:.4f}
✓ Testing R² Score: {test_r2:.4f}

Interpretation:
- The R² score of {test_r2:.4f} means the model explains {test_r2*100:.2f}% of variance
  in the target variable using the given features.
- RMSE of {test_rmse:.4f} indicates average prediction error.
- Lower residuals and higher R² indicate better model performance.

Key Features (Top 3 with highest influence):
""")
for idx, (_, row) in enumerate(coefficients_df.head(3).iterrows(), 1):
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {idx}. {row['Feature']}: {direction} heart disease (Coef: {row['Coefficient']:.4f})")

print("\n" + "="*60)
print("Multiple Linear Regression analysis completed successfully!")
print("="*60)
