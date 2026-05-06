# Sales Forecasting System

## Objective
Build an end-to-end time series forecasting system with REST API support.

## Technologies Used
- Python
- Pandas
- XGBoost
- ARIMA
- Prophet
- LSTM
- FastAPI
- Scikit-learn
- TensorFlow

## Features
- Data preprocessing
- Feature engineering
- Multiple forecasting models
- Model comparison
- REST API for predictions
- Swagger API documentation

## Models Implemented
1. ARIMA
2. Prophet
3. XGBoost
4. LSTM

## Best Model
LSTM achieved the lowest RMSE and performed best.

## API Endpoint
GET /predict

## How to Run

### Install libraries
```bash
pip install -r requirements.txt
```

### Run API
```bash
uvicorn app:app --reload
```

### Open Swagger Docs
http://127.0.0.1:8000/docs