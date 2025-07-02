import pandas as pd
import numpy as np
import joblib
import random
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set the number of time steps according to requirements
n_steps = 100

def add_technical_indicators(df, price_col='close'):
    """
    Add technical indicators to improve price prediction
    """
    print("  🔧 Adding technical indicators...")
    
    # Simple Moving Averages
    df['sma_5'] = df[price_col].rolling(window=5).mean()
    df['sma_10'] = df[price_col].rolling(window=10).mean()
    df['sma_20'] = df[price_col].rolling(window=20).mean()
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_5'] = df[price_col].ewm(span=5).mean()
    df['ema_10'] = df[price_col].ewm(span=10).mean()
    df['ema_20'] = df[price_col].ewm(span=20).mean()
    
    # Price position relative to moving averages
    df['price_vs_sma5'] = df[price_col] / df['sma_5'] - 1
    df['price_vs_sma10'] = df[price_col] / df['sma_10'] - 1
    df['price_vs_sma20'] = df[price_col] / df['sma_20'] - 1
    
    # Moving average crossovers
    df['sma5_vs_sma10'] = df['sma_5'] / df['sma_10'] - 1
    df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20'] - 1
    
    # Volatility indicators
    df['returns'] = df[price_col].pct_change()
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df[price_col].rolling(window=20).std()
    
    # Price momentum
    df['momentum_3'] = df[price_col] / df[price_col].shift(3) - 1
    df['momentum_5'] = df[price_col] / df[price_col].shift(5) - 1
    df['momentum_10'] = df[price_col] / df[price_col].shift(10) - 1
    
    # Rate of Change
    df['roc_5'] = df[price_col].pct_change(5)
    df['roc_10'] = df[price_col].pct_change(10)
    
    # Bollinger Bands
    df['bb_middle'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI approximation
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Price highs and lows
    df['high_5'] = df[price_col].rolling(window=5).max()
    df['low_5'] = df[price_col].rolling(window=5).min()
    df['high_10'] = df[price_col].rolling(window=10).max()
    df['low_10'] = df[price_col].rolling(window=10).min()
    
    # Distance from highs/lows
    df['dist_from_high5'] = (df['high_5'] - df[price_col]) / df[price_col]
    df['dist_from_low5'] = (df[price_col] - df['low_5']) / df[price_col]
    
    return df

def create_feature_sequences(data, feature_cols, target_col, n_steps):
    """
    Create sequences using multiple features for price prediction
    """
    X, y = [], []
    
    with tqdm(total=len(data) - n_steps, desc="Creating feature sequences", unit="sequences") as pbar:
        for i in range(n_steps, len(data)):
            # Get sequence of features
            sequence = data[feature_cols].iloc[i-n_steps:i].values.flatten()
            target = data[target_col].iloc[i]
            
            X.append(sequence)
            y.append(target)
            pbar.update(1)
    
    return np.array(X), np.array(y)

def calculate_direction_accuracy(y_true_price, y_pred_price, current_prices, threshold=0.0001):
    """
    Calculate direction accuracy by comparing actual and predicted price movements
    """
    actual_directions = []
    predicted_directions = []
    
    for i in range(len(y_true_price)):
        # Actual direction
        actual_change = (y_true_price[i] - current_prices[i]) / current_prices[i]
        actual_dir = 1 if actual_change > threshold else 0
        
        # Predicted direction
        pred_change = (y_pred_price[i] - current_prices[i]) / current_prices[i]
        pred_dir = 1 if pred_change > threshold else 0
        
        # Only count clear movements
        if abs(actual_change) > threshold:
            actual_directions.append(actual_dir)
            predicted_directions.append(pred_dir)
    
    if len(actual_directions) > 0:
        accuracy = np.mean(np.array(actual_directions) == np.array(predicted_directions))
        return accuracy, actual_directions, predicted_directions
    else:
        return 0.0, [], []

# Load data from CSV file
csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'

print(f"📈 Loading EURUSD data from CSV file: {csv_file}")

try:
    # Try to load the CSV file
    if os.path.exists(csv_file):
        print(f"Found CSV file: {csv_file}")
        
        # Load CSV data - this appears to be tab-separated MetaTrader export
        df = pd.read_csv(csv_file, sep='\t')
        
        print("CSV file structure:")
        print(f"Columns: {list(df.columns)}")
        
        # Check if it's the standard MetaTrader format with tab separators
        if '<DATE>' in df.columns and '<TIME>' in df.columns and '<CLOSE>' in df.columns:
            print("✅ Detected MetaTrader export format with tab separators")
            
            # Combine date and time columns
            df['datetime'] = df['<DATE>'] + ' ' + df['<TIME>']
            
            # Create the data DataFrame with proper columns
            data = pd.DataFrame()
            data['time'] = df['datetime']
            data['close'] = df['<CLOSE>']
            
            # Convert time column to datetime with MetaTrader format
            data['time'] = pd.to_datetime(data['time'], format='%Y.%m.%d %H:%M:%S')
            data = data.set_index('time')
            
            # Ensure close prices are numeric
            data['close'] = pd.to_numeric(data['close'], errors='coerce')
            data = data.dropna()
            
            print(f"✅ Successfully loaded CSV data with {len(data)} records")
            print(f"📅 Data range: {data.index[0]} to {data.index[-1]}")
            print(f"💰 Price range: {data['close'].min():.5f} to {data['close'].max():.5f}")
            
        else:
            raise ValueError("CSV format not recognized")
            
except Exception as e:
    print(f"❌ Error loading CSV file: {str(e)}")
    exit(1)

# Enhanced feature engineering
print("\n🔧 Step 1/5: Enhanced feature engineering...")
data = add_technical_indicators(data, 'close')

# Remove NaN values from technical indicators
data = data.dropna()
print(f"✅ Data after cleaning: {len(data)} records")

# Define feature columns (all technical indicators)
feature_cols = [col for col in data.columns if col not in ['close']]
print(f"📊 Using {len(feature_cols)} features: {feature_cols[:5]}... (showing first 5)")

# Split data
training_size = int(len(data) * 0.70)
train_data = data.iloc[:training_size]
test_data = data.iloc[training_size:]

print(f"\n📊 Data split:")
print(f"  🏋️ Training: {len(train_data)} records ({training_size/len(data)*100:.1f}%)")
print(f"  🧪 Testing: {len(test_data)} records ({(len(data)-training_size)/len(data)*100:.1f}%)")

# Create sequences for training
print("\n📦 Step 2/5: Creating feature sequences...")
X_train, y_train = create_feature_sequences(train_data, feature_cols, 'close', n_steps)
X_test, y_test = create_feature_sequences(test_data, feature_cols, 'close', n_steps)

print(f"✅ Training sequences: {X_train.shape}")
print(f"✅ Test sequences: {X_test.shape}")

# Create price regressor for price prediction
print("\n📈 Step 3/5: Training XGBoost Price Regressor...")

# Using XGBoost GPU directly for price regression
import xgboost as xgb
print("  🎮 Using XGBoost GPU for price regression!")
price_regressor = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',  # Updated method
        device='cuda',  # GPU acceleration (updated parameter)
        n_estimators=150,  # More trees for better accuracy
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    ))
])
print("  ✅ Using XGBoost GPU regressor!")

# Train price regressor
print(f"  🔄 Training XGBoost GPU Price Regressor (150 trees)...")
print(f"  ⏳ This may take 15-30 seconds with GPU acceleration...")
price_regressor.fit(X_train, y_train)
print("✅ Price regressor training completed!")

# Evaluate model
print("\n📋 Step 4/5: Evaluating model performance...")

# Price predictions
y_pred_price = price_regressor.predict(X_test)
price_mse = mean_squared_error(y_test, y_pred_price)
price_r2 = r2_score(y_test, y_pred_price)

# Calculate direction accuracy from price predictions
current_prices_test = test_data['close'].iloc[n_steps-1:-1].values
direction_accuracy, actual_dirs, pred_dirs = calculate_direction_accuracy(
    y_test, y_pred_price, current_prices_test
)

print(f"📊 Model Performance:")
print(f"  📉 Price MSE: {price_mse:.8f}")
print(f"  📈 Price R²: {price_r2:.4f}")
print(f"  🎯 Direction Accuracy (derived): {direction_accuracy:.4f} ({direction_accuracy*100:.2f}%)")

# Direction breakdown
if len(actual_dirs) > 0:
    actual_dirs = np.array(actual_dirs)
    pred_dirs = np.array(pred_dirs)
    
    print(f"\n📋 Direction Performance Breakdown:")
    for direction in [0, 1]:
        mask = actual_dirs == direction
        if np.sum(mask) > 0:
            acc = np.mean(pred_dirs[mask] == actual_dirs[mask]) * 100
            direction_name = {0: "DOWN", 1: "UP"}[direction]
            print(f"  📊 {direction_name} accuracy: {acc:.1f}% ({np.sum(mask)} samples)")

# Create visualizations
print("\n📊 Step 5/5: Creating performance visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Price predictions vs actual
n_plot = min(500, len(y_test))
axes[0,0].plot(y_test[:n_plot], label='Actual', alpha=0.7, color='blue')
axes[0,0].plot(y_pred_price[:n_plot], label='Predicted', alpha=0.7, color='red')
axes[0,0].set_title('Price Predictions vs Actual')
axes[0,0].set_xlabel('Time Steps')
axes[0,0].set_ylabel('Price')
axes[0,0].legend()
axes[0,0].grid(True)

# Prediction errors
errors = y_pred_price[:n_plot] - y_test[:n_plot]
axes[0,1].plot(errors, alpha=0.7, color='orange')
axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0,1].set_title('Price Prediction Errors')
axes[0,1].set_xlabel('Time Steps')
axes[0,1].set_ylabel('Error')
axes[0,1].grid(True)

# Direction accuracy over time (derived from price predictions)
if len(actual_dirs) > 0:
    direction_correct = (np.array(pred_dirs) == np.array(actual_dirs)).astype(int)
    rolling_dir_acc = pd.Series(direction_correct).rolling(window=20, min_periods=1).mean()
    axes[1,0].plot(rolling_dir_acc * 100, color='green', alpha=0.8)
    axes[1,0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    axes[1,0].set_title('Rolling Direction Accuracy (Derived from Price)')
    axes[1,0].set_xlabel('Time Steps')
    axes[1,0].set_ylabel('Accuracy (%)')
    axes[1,0].legend()
    axes[1,0].grid(True)

# Sample price data with indicators
sample_slice = data.tail(100)
axes[1,1].plot(sample_slice['close'], label='Close Price', color='blue')
axes[1,1].plot(sample_slice['sma_20'], label='SMA 20', color='orange', alpha=0.7)
axes[1,1].plot(sample_slice['ema_10'], label='EMA 10', color='green', alpha=0.7)
axes[1,1].fill_between(range(len(sample_slice)), 
                      sample_slice['bb_lower'], sample_slice['bb_upper'], 
                      alpha=0.2, color='gray', label='Bollinger Bands')
axes[1,1].set_title('Sample Data with Technical Indicators')
axes[1,1].set_xlabel('Time Steps')
axes[1,1].set_ylabel('Price')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('unified_price_model_performance.png', dpi=300, bbox_inches='tight')
print("📊 Performance visualization saved as 'unified_price_model_performance.png'")
plt.close()

# Save model
print("\n💾 Saving unified price model...")

os.makedirs('models', exist_ok=True)

# Save only the price model
joblib.dump(price_regressor, 'models/unified_price_regressor.pkl')

# Save model metadata
model_info = {
    'n_steps': n_steps,
    'features_used': feature_cols,
    'n_features': len(feature_cols),
    'training_size': len(X_train),
    'test_size': len(X_test),
    'price_mse': float(price_mse),
    'price_r2': float(price_r2),
    'direction_accuracy_derived': float(direction_accuracy),
    'model_type': 'Unified XGBoost Price Regressor with Derived Direction',
    'regressor_estimators': 150,
    'approach': 'single_model_price_based_direction'
}

import json
with open('models/unified_price_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"✅ Model saved:")
print(f"  📈 Unified price regressor: models/unified_price_regressor.pkl")
print(f"  📋 Model info: models/unified_price_model_info.json")

# Test unified prediction
print(f"\n🧪 Testing unified prediction...")
if len(X_test) > 0:
    sample_idx = 0
    sample_features = X_test[sample_idx:sample_idx+1]
    
    pred_price = price_regressor.predict(sample_features)[0]
    actual_price = y_test[sample_idx]
    current_price = current_prices_test[sample_idx]
    
    # Derive direction from price prediction
    pred_direction = 1 if pred_price > current_price else 0
    actual_direction = 1 if actual_price > current_price else 0
    
    direction_names = {0: "DOWN ↘", 1: "UP ↗"}
    
    print(f"🎯 Sample Unified Prediction:")
    print(f"  💰 Current Price: {current_price:.5f}")
    print(f"  📈 Predicted Price: {pred_price:.5f}")
    print(f"  📊 Derived Direction: {direction_names.get(pred_direction, 'Unknown')}")
    print(f"  ✅ Actual Price: {actual_price:.5f}")
    print(f"  ✅ Actual Direction: {direction_names.get(actual_direction, 'Unknown')}")
    
    direction_correct = "✅ CORRECT" if pred_direction == actual_direction else "❌ WRONG"
    price_error = abs(pred_price - actual_price)
    
    print(f"  🎯 Direction: {direction_correct}")
    print(f"  📏 Price Error: {price_error:.5f} ({price_error/actual_price*10000:.1f} pips)")

print(f"\n🎉 Unified model training completed!")
print(f"📈 Price prediction R²: {price_r2:.4f}")
print(f"🎯 Direction accuracy (derived): {direction_accuracy*100:.2f}%")
print(f"✨ Approach: Single price model with consistent direction derivation")
print(f"🚀 No more contradictory predictions!")
print(f"\n✨ Ready to use! Run the updated test script.") 