import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Read the dataset
df = pd.read_csv('water_data.csv')

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Define features (X) and target (y)
X = df_imputed[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                'Organic_carbon', 'Trihalomethanes', 'Turbidity']]
y = df_imputed['Potability']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=5)

# Create and train the logistic regression model
reg = LogisticRegression()
reg.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(reg, X_scaled, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Evaluate the model on the test set
test_score = reg.score(X_test, y_test)
print("Test Set Score:", test_score)

# Save the trained model
joblib.dump(reg, 'logistic_regression_model.pkl')
