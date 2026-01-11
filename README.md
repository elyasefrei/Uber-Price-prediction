Uber & Lyft Ride Price Prediction

Machine Learning Project

 Project Overview : 

This project aims to predict ride prices for Uber and Lyft services using machine learning techniques.
By combining ride information, temporal features, and weather data, we compare two regression models:

K-Nearest Neighbors (KNN) Regression

Linear Regression

The goal is to evaluate which model best explains and predicts ride prices, while providing insights into the most influential pricing factors.

Dataset :

Source:
Kaggle – Uber & Lyft Cab Prices Dataset
https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices

Files used:

cab_rides.csv → ride details (price, distance, cab type, locations, surge, time)

weather.csv → hourly weather conditions by location

Features Used :
Numerical Features

distance

surge_multiplier

hour_of_day

day_of_week

month

is_weekend

is_rush_hour

Categorical Features (Encoded)

cab_type (Uber / Lyft)

name (ride service)

source (pickup location)

destination

time_period (Morning / Afternoon / Evening / Night)

distance_category (Short / Medium / Long / Very_Long)

Weather Features

temp

clouds

pressure

rain

humidity

wind

Target Variable: price (USD)

 Data Cleaning & Preprocessing : 

Removed missing target values (price)

Filled missing surge multipliers with 1.0

Removed rows with missing critical features

Outlier removal using IQR method

Feature scaling with StandardScaler

Categorical encoding with LabelEncoder

Train/Test split: 80% / 20%

Feature Engineering :

Time-based features from timestamp

Weekend & rush hour indicators

Distance categorization

Hourly aggregation and merge with weather data

Exploratory Data Analysis (EDA) :

Key analyses include:

Price distribution

Price vs distance relationship

Average price by cab type and location

Impact of outliers on price distribution

All plots are saved in the visualizations/ folder.

 Models & Training :
K-Nearest Neighbors (KNN)

Optimized number of neighbors (k)

Best k selected using test R² score

Non-parametric, distance-based model

Linear Regression

Ordinary Least Squares (OLS)

Provides interpretability via coefficients

Used for feature importance analysis

Model Evaluation Metrics

R² Score

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

5-Fold Cross-Validation

Model Comparison (Test Set)
Metric	KNN	Linear Regression
R² Score	Higher	Lower
RMSE	Lower	Higher
MAE	Lower	Higher
Interpretability :

Best Overall Model: K-Nearest Neighbors (KNN)
The KNN model explains the highest proportion of variance and produces the lowest prediction errors.

Feature Importance (Linear Regression)

Top drivers of price:

Distance

Surge multiplier

Ride service type / location-related features

A full ranked list is saved in:

results_feature_importance.csv

Project Structure:
.
├── cab_rides.csv
├── weather.csv
├── README.md
├── results_model_comparison.csv
├── results_feature_importance.csv
├── results_sample_predictions.csv
├── visualizations/
│   ├── 01_eda_price_analysis.png
│   ├── 02_knn_k_optimization.png
│   ├── 03_model_comparison.png
│   ├── 04_actual_vs_predicted.png
│   ├── 05_residual_analysis.png
│   └── 06_feature_importance.png
└── main_notebook.py

 Key Insights : 

Ride price is strongly influenced by distance, surge pricing, and time-related factors

Weather has a measurable but secondary impact

KNN captures non-linear pricing patterns better than Linear Regression

Linear Regression remains valuable for interpretability

Practical Applications :

Fare estimation for users before booking

Pricing optimization for ride-sharing platforms

Decision support for dynamic pricing strategies
Technologies Used :

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn
