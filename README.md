# Gold & Silver Price Prediction using Machine Learning and Deep Learning

A comprehensive time-series forecasting project for predicting **Gold and Silver prices** using a combination of statistical, machine learning, and deep learning approaches.

The project compares **LSTM, ARIMA, Random Forest, and Gradient Boosting** models and uses advanced financial/technical features such as moving averages, exponential moving averages, RSI, MACD, Bollinger Bands, volatility, momentum, rate of change, lag features, and time-based features.

---

## 📌 Project Overview

Gold and silver are highly dynamic financial commodities whose prices are influenced by historical trends, volatility, market momentum, and other time-dependent patterns.

This project develops a multi-model forecasting framework to analyze historical Gold and Silver price data and generate future price predictions.

The system implements three major forecasting approaches:

1. **LSTM (Long Short-Term Memory)** — Deep learning model for sequential time-series prediction.
2. **ARIMA (AutoRegressive Integrated Moving Average)** — Statistical time-series forecasting model.
3. **Ensemble Model** — Combines Random Forest and Gradient Boosting predictions.

The models are evaluated using multiple regression metrics, and the results are visualized through actual-vs-predicted plots and training-history plots.

---

## 🎯 Problem Statement

Accurately forecasting precious-metal prices is challenging because financial time-series data contains trends, volatility, temporal dependencies, and nonlinear patterns.

Traditional statistical models can capture temporal dependencies effectively, while machine learning and deep learning models can capture complex nonlinear relationships.

Therefore, this project aims to develop and compare multiple forecasting approaches for Gold and Silver price prediction and determine their effectiveness using standard regression evaluation metrics.

---

## 🎯 Objectives

* Analyze historical Gold and Silver price data.
* Perform data cleaning and preprocessing.
* Engineer advanced technical indicators.
* Develop an LSTM-based deep learning forecasting model.
* Develop an ARIMA-based statistical forecasting model.
* Develop a machine learning ensemble using Random Forest and Gradient Boosting.
* Compare model performance using multiple evaluation metrics.
* Visualize actual and predicted prices.
* Generate long-term forecasts using ARIMA.
* Provide a reusable framework for precious-metal price forecasting.

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────────────┐
                    │   Historical Price Data  │
                    │                          │
                    │   Gold Prices             │
                    │   Silver Prices           │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Data Preprocessing        │
                    │                          │
                    │ • Column Cleaning         │
                    │ • Numeric Conversion      │
                    │ • Date Processing         │
                    │ • Missing Value Handling  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Feature Engineering       │
                    │                          │
                    │ • Moving Averages         │
                    │ • EMA                     │
                    │ • RSI                     │
                    │ • MACD                    │
                    │ • Bollinger Bands         │
                    │ • Volatility              │
                    │ • Momentum                │
                    │ • ROC                     │
                    │ • Lag Features            │
                    │ • Time Features           │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
       │    LSTM     │    │    ARIMA    │    │ Ensemble Model  │
       │             │    │             │    │                 │
       │ Deep        │    │ Statistical │    │ Random Forest   │
       │ Learning    │    │ Forecasting │    │       +         │
       │             │    │             │    │ Gradient Boost  │
       └──────┬──────┘    └──────┬──────┘    └────────┬────────┘
              │                  │                    │
              └──────────────────┼────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Model Evaluation          │
                    │                          │
                    │ • MSE                    │
                    │ • RMSE                   │
                    │ • MAE                    │
                    │ • R² Score                │
                    │ • MAPE                   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Visualization & Forecast │
                    │                          │
                    │ • Prediction Plots       │
                    │ • Training History       │
                    │ • Future Forecasts       │
                    └──────────────────────────┘
```

---

## 🔄 Workflow

The complete workflow consists of the following stages:

### 1. Data Acquisition

Historical Gold and Silver price datasets are loaded from CSV files.

The datasets contain time-series information such as:

* Date
* Close/Last Price
* Open
* High
* Low
* Volume

---

### 2. Data Preprocessing

The preprocessing stage performs:

* Column-name cleaning.
* Conversion of `Close/Last` to `Price`.
* Conversion of `Volume` to `Vol.`.
* Removal of commas from numerical values.
* Conversion of numerical columns into numeric data types.
* Date conversion.
* Chronological sorting.
* Missing-value handling.
* Date indexing.

The data is sorted chronologically before being passed to the forecasting models.

---

### 3. Advanced Feature Engineering

The project creates several technical indicators and time-series features.

#### Price Returns

Percentage change in price is calculated to capture short-term price movement.

#### Log Returns

Logarithmic returns are calculated to represent continuous price changes.

#### Moving Averages

The project calculates:

```text
MA_5
MA_10
MA_20
MA_50
MA_100
MA_200
```

Moving averages help identify price trends over different time windows.

#### Moving Average Ratios

The relationship between the current price and moving averages is also calculated.

```text
Price / Moving Average
```

#### Exponential Moving Averages

The project calculates:

```text
EMA_12
EMA_26
EMA_50
```

EMA gives greater importance to recent observations.

#### Volatility

Rolling volatility is calculated over:

```text
10-day window
30-day window
```

#### RSI

The Relative Strength Index is calculated using a 14-period window.

RSI can provide information about the relative strength of recent upward and downward price movements.

#### MACD

The Moving Average Convergence Divergence indicator is calculated using:

```text
12-period EMA
26-period EMA
9-period signal EMA
```

The project derives:

```text
MACD
MACD_signal
MACD_diff
```

#### Bollinger Bands

The project calculates:

```text
BB_middle
BB_upper
BB_lower
BB_width
BB_position
```

These features provide information about price volatility and the position of the current price relative to its rolling range.

#### Momentum

Momentum is calculated for:

```text
5 periods
10 periods
20 periods
```

#### Rate of Change

ROC is also calculated for:

```text
5 periods
10 periods
20 periods
```

#### High-Low Features

Where High and Low prices are available, the project calculates:

```text
HL_range
HL_ratio
Price_position
```

#### Volume Features

When volume data is available:

```text
Volume_MA_10
Volume_ratio
```

are calculated.

#### Lag Features

Historical price values are incorporated using lag features:

```text
Price_lag_1
Price_lag_2
Price_lag_3
Price_lag_5
Price_lag_7
Price_lag_10
Price_lag_15
Price_lag_20
```

#### Time Features

The following calendar features are included:

```text
DayOfWeek
Month
Quarter
DayOfMonth
```

---

# 🤖 Models

## 1. LSTM

The project uses a Bidirectional LSTM architecture for sequential price prediction.

### Architecture

```text
Input Sequence
      │
      ▼
Bidirectional LSTM (128 units)
      │
      ▼
Dropout (0.30)
      │
      ▼
Bidirectional LSTM (64 units)
      │
      ▼
Dropout (0.30)
      │
      ▼
LSTM (32 units)
      │
      ▼
Dropout (0.20)
      │
      ▼
Dense (16, ReLU)
      │
      ▼
Dense (1)
      │
      ▼
Predicted Price
```

The model uses:

* Min-Max Scaling
* Sequence length of 60
* Adam optimizer
* Learning rate of 0.001
* Huber loss
* Mean Absolute Error as an additional metric
* Early stopping
* Learning-rate reduction

The model is trained using an 80/20 chronological train-test split.

---

## 2. ARIMA

ARIMA is used as a statistical baseline and long-term forecasting model.

The implementation uses:

```text
ARIMA(5, 1, 2)
```

where:

* `p = 5` — autoregressive component
* `d = 1` — differencing order
* `q = 2` — moving-average component

The model is trained on 80% of the historical price data and evaluated on the remaining 20%.

The project also generates a **730-step forecast**, representing approximately two years of future daily predictions.

---

## 3. Random Forest

Random Forest is implemented as one component of the ensemble forecasting approach.

Configuration:

```text
n_estimators = 200
max_depth = 15
random_state = 42
```

Random Forest is useful for modeling nonlinear relationships between engineered technical indicators and price.

---

## 4. Gradient Boosting

Gradient Boosting is used as the second component of the ensemble.

Configuration:

```text
n_estimators = 200
max_depth = 5
learning_rate = 0.05
random_state = 42
```

Gradient Boosting builds models sequentially to improve prediction performance.

---

## 5. Ensemble Model

The project combines Random Forest and Gradient Boosting predictions using a weighted average:

```text
Ensemble Prediction =
0.5 × Random Forest Prediction
+
0.5 × Gradient Boosting Prediction
```

This approach combines the predictions of two different machine-learning algorithms.

---

# 📊 Model Evaluation

The models are evaluated using multiple regression metrics.

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

Lower MSE indicates better performance.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and represents prediction error in the same unit as the target variable.

Lower RMSE is better.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted prices.

Lower MAE indicates better performance.

### R² Score

R² measures how well the model explains the variation in the target variable.

A value closer to 1 generally indicates stronger explanatory performance.

### Mean Absolute Percentage Error (MAPE)

MAPE measures prediction error as a percentage.

Lower MAPE indicates better forecasting performance.

### Accuracy-like Score

The implementation also prints:

```text
100 - MAPE
```

as an accuracy-like percentage.

> **Note:** This should not be interpreted as classification accuracy. This project is a regression/time-series forecasting problem, so MAPE, MAE, RMSE, and R² are more appropriate metrics for evaluating performance.

---

# 📈 Visualizations

The project generates several visualizations.

## Actual vs Predicted Prices

Prediction plots compare:

```text
Actual Price
vs
Predicted Price
```

for:

* Gold — LSTM
* Silver — LSTM
* Gold — ARIMA
* Silver — ARIMA
* Gold — Ensemble
* Silver — Ensemble

---

## LSTM Training History

The project visualizes:

```text
Training Loss
Validation Loss
Training MAE
Validation MAE
```

This helps analyze model convergence and potential overfitting.

---

# 🔮 Future Forecasting

The ARIMA model generates a **730-step future forecast** for both Gold and Silver.

The final forecast is saved as:

```text
gold_forecast.csv
silver_forecast.csv
```

The forecast output contains:

```text
Day
Predicted_Price
```

---

# 📁 Project Structure

```text
gold-silver-price-prediction/
│
├── price_prediction.py
│
├── gold prices.csv
├── silver prices.csv
│
├── gold_forecast.csv
├── silver_forecast.csv
│
├── requirements.txt
├── README.md
│
└── .gitignore
```

---

# 🛠️ Technologies Used

### Programming Language

* Python 3.11

### Data Processing

* NumPy
* Pandas

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Deep Learning

* TensorFlow
* Keras

### Time-Series Analysis

* Statsmodels
* ARIMA

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/gold-silver-price-prediction.git
```

Navigate into the project:

```bash
cd gold-silver-price-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Make sure the following datasets are located in the project directory:

```text
gold prices.csv
silver prices.csv
```

Then run:

```bash
python price_prediction.py
```

The program will:

```text
1. Load Gold and Silver datasets
2. Preprocess the data
3. Generate technical indicators
4. Train the LSTM models
5. Train ARIMA models
6. Train Random Forest
7. Train Gradient Boosting
8. Generate ensemble predictions
9. Calculate evaluation metrics
10. Generate visualizations
11. Generate 730-step ARIMA forecasts
12. Save forecast results
```

---

# 📋 Example Output

The program displays model performance in the terminal in the following format:

```text
============================================================
LSTM MODEL PERFORMANCE
============================================================
MSE:  ...
RMSE: ...
MAE:  ...
R² Score: ...
MAPE: ...%
Accuracy: ...%
```

Similar performance summaries are generated for ARIMA and the ensemble model.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical implementation of:

* Time-series forecasting
* Supervised learning
* Deep learning
* LSTM networks
* Bidirectional LSTM
* Statistical forecasting
* ARIMA
* Ensemble learning
* Random Forest
* Gradient Boosting
* Feature engineering
* Technical indicators
* Data normalization
* Sequence generation
* Train-test splitting
* Model evaluation
* Forecasting
* Data visualization

---

# 🔬 Technical Highlights

### Multi-model forecasting

Instead of relying on a single forecasting algorithm, the project compares statistical, machine-learning, and deep-learning approaches.

### Feature-rich modeling

The machine-learning and LSTM pipelines use a wide range of engineered time-series and technical features.

### Sequential learning

The LSTM model uses sequences of historical observations to learn temporal dependencies.

### Statistical baseline

ARIMA provides a classical statistical forecasting approach against which other models can be compared.

### Ensemble learning

Random Forest and Gradient Boosting predictions are combined to create an ensemble forecast.

### Long-term forecasting

The ARIMA implementation generates a 730-step future forecast for both commodities.

---

# ⚠️ Limitations

This project is intended for educational and research purposes and should not be considered financial advice.

Important limitations include:

* Historical price patterns do not guarantee future performance.
* Commodity prices are influenced by many external factors.
* The current models primarily rely on historical price and derived technical features.
* Macroeconomic indicators are not included in the current implementation.
* News and sentiment information are not incorporated.
* Geopolitical events are not explicitly modeled.
* Long-term forecasts may become increasingly uncertain as the forecast horizon increases.

---

# 🚀 Future Improvements

Potential improvements include:

* Hyperparameter optimization.
* Automated ARIMA parameter selection.
* Incorporating macroeconomic indicators.
* Adding USD/INR exchange-rate features.
* Adding crude oil and stock-market indicators.
* Incorporating financial news sentiment.
* Using Transformer-based time-series models.
* Implementing Temporal Fusion Transformers.
* Implementing XGBoost and LightGBM.
* Developing hybrid LSTM-ARIMA models.
* Using walk-forward validation.
* Performing cross-validation specifically designed for time-series data.
* Building an interactive Streamlit dashboard.
* Deploying the forecasting system as a web application.
* Adding real-time market-data integration.
* Adding confidence intervals to future forecasts.

---

# 📌 Important Note About Forecasting

Predictions generated by this project are model-based estimates derived from historical data.

They should not be interpreted as guaranteed future Gold or Silver prices.

This repository is intended to demonstrate the application of **Machine Learning, Deep Learning, Statistical Modeling, and Time-Series Forecasting** to financial commodity data.

---

# 👨‍💻 Author

**Abdul Basith P**

M.Sc. Computer Science — Artificial Intelligence & Machine Learning

Rajiv Gandhi National Institute of Youth Development (RGNIYD)

---

# ⭐ Project Highlights for Recruiters

This project demonstrates practical experience in:

```text
Python
Machine Learning
Deep Learning
Time-Series Forecasting
LSTM
ARIMA
Random Forest
Gradient Boosting
Feature Engineering
Technical Indicators
Scikit-learn
TensorFlow/Keras
Pandas
NumPy
Matplotlib
Seaborn
Statistical Modeling
Model Evaluation
```

---

# 📄 License

This project is intended for educational and research purposes.
