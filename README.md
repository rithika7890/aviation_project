# Aircraft Engine Remaining Useful Life (RUL) Prediction
## Overview
This project focuses on predicting the Remaining Useful Life (RUL) of aircraft engines using Machine Learning techniques and the NASA CMAPSS dataset. The system helps estimate the number of operational cycles remaining before engine failure, enabling predictive maintenance and reducing unexpected downtime.
The project combines data preprocessing, feature engineering, degradation modelling, and XGBoost supervised learning to create an intelligent and explainable predictive maintenance framework.

## Features
- Aircraft engine Remaining Useful Life (RUL) prediction
- Predictive maintenance analysis
- Feature engineering and rolling statistical feature generation
- Engine degradation modelling
- Data preprocessing and target capping
- Explainable machine learning workflow
- Real-time dashboard visualization support

## Tech Stack
- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- NASA CMAPSS Dataset

## Dataset
The project uses the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset for aircraft engine degradation analysis and RUL prediction.
Dataset includes:
- Sensor readings
- Operational settings
- Engine cycle information
- Failure progression patterns

## Project Workflow
1. Data collection from NASA CMAPSS dataset
2. Data cleaning and preprocessing
3. Feature engineering and rolling statistical analysis
4. Target capping for model optimization
5. Training XGBoost regression model
6. Predicting Remaining Useful Life (RUL)
7. Performance evaluation using MAE, RMSE, and R² metrics

## Results
Model Performance:
- MAE: 12.829
- RMSE: 17.897
- R² Score: 0.831
The model successfully predicted aircraft engine degradation trends and demonstrated strong predictive maintenance capabilities.

