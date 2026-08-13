import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 6)

# ============================================
# ENHANCED DATA PREPROCESSING
# ============================================

def preprocess_data(data):
    """Enhanced preprocessing with better error handling"""
    data = data.copy()
    data.columns = data.columns.str.strip()

    # Rename columns
    data.rename(columns={'Close/Last': 'Price', 'Volume': 'Vol.'}, inplace=True)

    if 'Date' not in data.columns:
        raise ValueError("Date column not found!")

    # Clean and convert numeric columns
    for col in ['Price', 'Open', 'High', 'Low']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '')
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle Volume
    if 'Vol.' in data.columns:
        data['Vol.'] = data['Vol.'].astype(str).str.replace(',', '')
        data['Vol.'] = data['Vol.'].apply(
            lambda x: float(x.replace('K', '')) * 1000 if 'K' in str(x)
            else (float(x) if x not in ['N/A', '-', 'nan'] else np.nan)
        )

    # Date handling
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.sort_values('Date')
    data = data.dropna(subset=['Date', 'Price'])
    data.set_index('Date', inplace=True)

    # Forward fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')

    return data


# ============================================
# ADVANCED FEATURE ENGINEERING
# ============================================

def create_advanced_features(data):
    """Create comprehensive technical indicators"""
    df = data.copy()

    # Price-based features
    df['Returns'] = df['Price'].pct_change()
    df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))

    # Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{window}'] = df['Price'].rolling(window=window).mean()
        df[f'MA_ratio_{window}'] = df['Price'] / df[f'MA_{window}']

    # Exponential Moving Averages
    for span in [12, 26, 50]:
        df[f'EMA_{span}'] = df['Price'].ewm(span=span, adjust=False).mean()

    # Volatility
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Volatility_30'] = df['Returns'].rolling(window=30).std()

    # RSI (Relative Strength Index)
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Price'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['BB_middle'] = df['Price'].rolling(window=20).mean()
    bb_std = df['Price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (df['Price'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Price Momentum
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Price'] - df['Price'].shift(period)
        df[f'ROC_{period}'] = df['Price'].pct_change(periods=period)

    # High-Low features
    if 'High' in df.columns and 'Low' in df.columns:
        df['HL_range'] = df['High'] - df['Low']
        df['HL_ratio'] = df['High'] / df['Low']
        df['Price_position'] = (df['Price'] - df['Low']) / (df['High'] - df['Low'])

    # Volume features
    if 'Vol.' in df.columns:
        df['Volume_MA_10'] = df['Vol.'].rolling(window=10).mean()
        df['Volume_ratio'] = df['Vol.'] / df['Volume_MA_10']

    # Lag features
    for lag in [1, 2, 3, 5, 7, 10, 15, 20]:
        df[f'Price_lag_{lag}'] = df['Price'].shift(lag)

    # Time features
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['DayOfMonth'] = df.index.day

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ============================================
# ENHANCED LSTM MODEL
# ============================================

def create_sequences(data, seq_length, target_col='Price'):
    """Create sequences for LSTM with multiple features"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data.iloc[i + seq_length, data.columns.get_loc(target_col)])
    return np.array(X), np.array(y)


def build_advanced_lstm(input_shape):
    """Build improved LSTM architecture"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

    return model


def train_lstm_model(data, seq_length=60, epochs=50, batch_size=32):
    """Train enhanced LSTM model"""
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)

    # Select features
    feature_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Vol.']]
    df_features = data[feature_cols].copy()
    df_features = df_features.dropna()

    print(f"Using {len(feature_cols)} features")
    print(f"Data shape: {df_features.shape}")

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features.values)
    scaled_df = pd.DataFrame(scaled_data, columns=df_features.columns, index=df_features.index)

    # Create sequences
    X, y = create_sequences(scaled_df, seq_length, target_col='Price')

    # Train-test split (80-20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Build model
    model = build_advanced_lstm((X_train.shape[1], X_train.shape[2]))

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Predict
    y_pred_scaled = model.predict(X_test)

    # Inverse transform predictions
    dummy = np.zeros((len(y_pred_scaled), scaled_df.shape[1]))
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, 0]

    dummy_test = np.zeros((len(y_test), scaled_df.shape[1]))
    dummy_test[:, 0] = y_test
    y_test_actual = scaler.inverse_transform(dummy_test)[:, 0]

    # Calculate metrics
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100

    print("\n" + "="*60)
    print("LSTM MODEL PERFORMANCE")
    print("="*60)
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Accuracy: {100 - mape:.2f}%")

    return model, scaler, df_features, y_test_actual, y_pred, history


# ============================================
# IMPROVED ARIMA MODEL
# ============================================

def train_arima_model(data, order=(5, 1, 2)):
    """Train ARIMA model with auto parameter selection"""
    print("\n" + "="*60)
    print("TRAINING ARIMA MODEL")
    print("="*60)

    prices = data['Price'].dropna()

    # Split data
    train_size = int(len(prices) * 0.8)
    train, test = prices[:train_size], prices[train_size:]

    print(f"Training ARIMA{order}...")
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test))

    # Calculate metrics
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    r2 = r2_score(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    print("\n" + "="*60)
    print("ARIMA MODEL PERFORMANCE")
    print("="*60)
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Accuracy: {100 - mape:.2f}%")

    # Forecast next 730 days (2 years)
    future_forecast = model_fit.forecast(steps=730)

    return model_fit, test, predictions, future_forecast


# ============================================
# ENSEMBLE MODEL
# ============================================

def train_ensemble_model(data):
    """Train Random Forest and Gradient Boosting ensemble"""
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MODEL (RF + GB)")
    print("="*60)

    # Create features
    df_features = create_advanced_features(data)
    df_features = df_features.dropna()

    # Prepare data
    feature_cols = [col for col in df_features.columns if col not in ['Price', 'Open', 'High', 'Low', 'Vol.']]
    X = df_features[feature_cols]
    y = df_features['Price']

    # Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)

    print("Training Random Forest...")
    rf.fit(X_train_scaled, y_train)

    print("Training Gradient Boosting...")
    gb.fit(X_train_scaled, y_train)

    # Predictions
    rf_pred = rf.predict(X_test_scaled)
    gb_pred = gb.predict(X_test_scaled)

    # Ensemble (weighted average)
    ensemble_pred = 0.5 * rf_pred + 0.5 * gb_pred

    # Metrics
    mse = mean_squared_error(y_test, ensemble_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100

    print("\n" + "="*60)
    print("ENSEMBLE MODEL PERFORMANCE")
    print("="*60)
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Accuracy: {100 - mape:.2f}%")

    return rf, gb, scaler, feature_cols, y_test, ensemble_pred


# ============================================
# VISUALIZATION
# ============================================

def plot_predictions(y_test, y_pred, title, color='orange'):
    """Plot actual vs predicted prices"""
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(y_test)), y_test, label='Actual', linewidth=2)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color=color, linewidth=2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution pipeline"""

    # Load data
    print("Loading data...")
    gold_data = pd.read_csv('gold prices.csv')
    silver_data = pd.read_csv('silver prices.csv')
    # Preprocess
    print("\nPreprocessing data...")
    gold_data = preprocess_data(gold_data)
    silver_data = preprocess_data(silver_data)

    print(f"\nGold data shape: {gold_data.shape}")
    print(f"Silver data shape: {silver_data.shape}")

    # Create features
    print("\nCreating advanced features...")
    gold_features = create_advanced_features(gold_data)
    silver_features = create_advanced_features(silver_data)

    # ==================== GOLD MODELS ====================
    print("\n" + "="*70)
    print("TRAINING GOLD PREDICTION MODELS")
    print("="*70)

    # LSTM
    gold_lstm, gold_scaler_lstm, gold_feat, gold_y_test_lstm, gold_y_pred_lstm, gold_history = \
        train_lstm_model(gold_features, seq_length=60, epochs=50)

    # ARIMA
    gold_arima, gold_test_arima, gold_pred_arima, gold_forecast = \
        train_arima_model(gold_data)

    # Ensemble
    gold_rf, gold_gb, gold_scaler_ens, gold_feat_cols, gold_y_test_ens, gold_y_pred_ens = \
        train_ensemble_model(gold_data)

    # ==================== SILVER MODELS ====================
    print("\n" + "="*70)
    print("TRAINING SILVER PREDICTION MODELS")
    print("="*70)

    # LSTM
    silver_lstm, silver_scaler_lstm, silver_feat, silver_y_test_lstm, silver_y_pred_lstm, silver_history = \
        train_lstm_model(silver_features, seq_length=60, epochs=50)

    # ARIMA
    silver_arima, silver_test_arima, silver_pred_arima, silver_forecast = \
        train_arima_model(silver_data)

    # Ensemble
    silver_rf, silver_gb, silver_scaler_ens, silver_feat_cols, silver_y_test_ens, silver_y_pred_ens = \
        train_ensemble_model(silver_data)

    # ==================== VISUALIZATIONS ====================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # LSTM predictions
    plot_predictions(gold_y_test_lstm, gold_y_pred_lstm,
                    'Gold Price Predictions (LSTM)', color='gold')
    plot_predictions(silver_y_test_lstm, silver_y_pred_lstm,
                    'Silver Price Predictions (LSTM)', color='silver')

    # Training history
    plot_training_history(gold_history)
    plot_training_history(silver_history)

    # ARIMA predictions
    plot_predictions(gold_test_arima.values, gold_pred_arima,
                    'Gold Price Predictions (ARIMA)', color='blue')
    plot_predictions(silver_test_arima.values, silver_pred_arima,
                    'Silver Price Predictions (ARIMA)', color='purple')

    # Ensemble predictions
    plot_predictions(gold_y_test_ens.values, gold_y_pred_ens,
                    'Gold Price Predictions (Ensemble)', color='green')
    plot_predictions(silver_y_test_ens.values, silver_y_pred_ens,
                    'Silver Price Predictions (Ensemble)', color='red')

    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("FINAL SUMMARY - 2 YEAR FORECASTS")
    print("="*70)

    print(f"\nGold - Current Price: ${gold_data['Price'].iloc[-1]:.2f}")
    print(f"Gold - ARIMA 2-Year Forecast: ${gold_forecast.iloc[-1]:.2f}")
    print(f"Gold - Expected Change: {((gold_forecast.iloc[-1] / gold_data['Price'].iloc[-1]) - 1) * 100:.2f}%")

    print(f"\nSilver - Current Price: ${silver_data['Price'].iloc[-1]:.2f}")
    print(f"Silver - ARIMA 2-Year Forecast: ${silver_forecast.iloc[-1]:.2f}")
    print(f"Silver - Expected Change: {((silver_forecast.iloc[-1] / silver_data['Price'].iloc[-1]) - 1) * 100:.2f}%")

    # Save forecasts
    gold_forecast_df = pd.DataFrame({
        'Day': range(1, 731),
        'Predicted_Price': gold_forecast
    })
    silver_forecast_df = pd.DataFrame({
        'Day': range(1, 731),
        'Predicted_Price': silver_forecast
    })

    gold_forecast_df.to_csv('gold_forecast.csv', index=False)
    silver_forecast_df.to_csv('silver_forecast.csv', index=False)

    print("\n✓ Forecasts saved to CSV files")
    print("\nAll models trained successfully!")


if __name__ == "__main__":
    main()