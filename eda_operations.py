import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Ensure the dataset file is in the same directory
try:
    df = pd.read_csv('Heart_Disease_Prediction.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Heart_Disease_Prediction.csv not found.")
    exit()

# ---------------------------------------------------------
# Step 1: Data Understanding
# ---------------------------------------------------------
print("\n--- First 5 rows of the dataset ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# ---------------------------------------------------------
# Step 2: Checking for Missing Values
# ---------------------------------------------------------
print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())

# ---------------------------------------------------------
# Step 3: Exploratory Data Analysis (EDA) - Visualizations
# ---------------------------------------------------------

# Set the style for the plots
sns.set_theme(style="whitegrid")

# 1. Distribution of Heart Disease (Target Variable)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Heart Disease', palette='viridis')
plt.title('Distribution of Heart Disease Presence vs Absence')
plt.xlabel('Heart Disease Status')
plt.ylabel('Count')
plt.show()

# Interpretation: This plot shows how many people in our dataset 
# have heart disease versus those who do not.

# 2. Histogram for Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='blue')
plt.title('Distribution of Patients Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Interpretation: Helps us understand the age range of 
# patients studied in this dataset.

# 3. Correlation Heatmap
# We need to convert categorical 'Heart Disease' to numeric for correlation
df_encoded = df.copy()
df_encoded['Heart Disease'] = df_encoded['Heart Disease'].map({'Presence': 1, 'Absence': 0})

plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# Interpretation: This heatmap shows how different medical factors 
# relate to each other and to the presence of heart disease.

# 4. Age vs Max Heart Rate Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Max HR', hue='Heart Disease', palette='magma')
plt.title('Age vs Max Heart Rate (Colored by Heart Disease)')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.show()

# Interpretation: This plot explores the relationship between 
# age and cardiovascular performance.

print("\nEDA operations completed.")
