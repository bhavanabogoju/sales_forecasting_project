import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import joblib

from preprocessing import load_and_clean_data
from feature_engineering import create_features


# Load data
df = load_and_clean_data("../data/Forecasting Case- Study.xlsx")


# Create features
df = create_features(df)


# Input features
X = df[['month', 'week', 'lag_1', 'lag_7', 'rolling_mean']]


# Target column
y = df['Total']


# Split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


# Create model
model = XGBRegressor()


# Train model
model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Calculate RMSE
rmse = mean_squared_error(y_test, predictions)

print("Model Trained Successfully!")

print("RMSE:", rmse)


# Save model
joblib.dump(model, "../models/xgboost_model.pkl")

print("Model Saved Successfully!")