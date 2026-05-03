import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
try:
    df = pd.read_csv('Heart_Disease_Prediction.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Heart_Disease_Prediction.csv not found.")
    exit()

# ---------------------------------------------------------
# Simple Linear Regression: Predicting Max HR based on Age
# ---------------------------------------------------------

# Selecting independent (X) and dependent (y) variables
# X must be a 2D array for scikit-learn
X = df[['Age']] 
y = df['Max HR']

print("\n--- Model Training ---")
# Split the data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# ---------------------------------------------------------
# Show Outputs and Model Evaluation
# ---------------------------------------------------------
print("\n--- Coefficients ---")
print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

print("\n--- Model Performance ---")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# ---------------------------------------------------------
# Visualizing the Results
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
# Plotting the actual data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
# Plotting the regression line
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Regression Line')

plt.title('Simple Linear Regression: Age vs Max Heart Rate')
plt.xlabel('Age')
plt.ylabel('Max HR')
plt.legend()
plt.show()

# ---------------------------------------------------------
# Interpretation of Results
# ---------------------------------------------------------
print("\n--- Interpretation ---")
print("1. The negative coefficient indicates that as Age increases, Max Heart Rate tends to decrease.")
print(f"2. R-squared value of {r2:.2f} shows how much variance in Max HR is explained by Age.")
print("3. Linear regression is useful here to find a general trend in cardiovascular health over time.")
