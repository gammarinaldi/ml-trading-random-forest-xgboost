import pandas as pd
import numpy as np
import joblib
import random
import os
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated XGBoost imports
try:
    import xgboost as xgb
    
    # Check if GPU is available
    try:
        # Try to create a GPU-enabled DMatrix to test GPU availability
        test_data = xgb.DMatrix(np.random.random((10, 5)))
        gpu_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        xgb.train(gpu_params, test_data, num_boost_round=1, verbose_eval=False)
        GPU_AVAILABLE = True
        print("✅ XGBoost GPU acceleration available!")
    except Exception:
        GPU_AVAILABLE = False
        print("⚠️ XGBoost GPU not available, using CPU")
        
except ImportError:
    print("❌ XGBoost not installed. Install with: pip install xgboost")
    exit(1)

# Standard imports
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set the number of time steps according to requirements
n_steps = 100

def add_technical_indicators(df, price_col='close'):
    """
    Add technical indicators for XGBoost training
    """
    print("  🔧 Adding technical indicators for XGBoost...")
    
    # Simple Moving Averages
    df['sma_5'] = df[price_col].rolling(window=5).mean()
    df['sma_10'] = df[price_col].rolling(window=10).mean()
    df['sma_20'] = df[price_col].rolling(window=20).mean()
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_5'] = df[price_col].ewm(span=5).mean()
    df['ema_10'] = df[price_col].ewm(span=10).mean()
    df['ema_20'] = df[price_col].ewm(span=20).mean()
    
    # Price ratios (very important for XGBoost)
    df['price_vs_sma5'] = df[price_col] / df['sma_5']
    df['price_vs_sma10'] = df[price_col] / df['sma_10']
    df['price_vs_sma20'] = df[price_col] / df['sma_20']
    df['price_vs_sma50'] = df[price_col] / df['sma_50']
    
    # Moving average crossovers
    df['sma5_vs_sma10'] = df['sma_5'] / df['sma_10']
    df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20']
    df['ema5_vs_ema10'] = df['ema_5'] / df['ema_10']
    
    # Volatility and momentum
    df['returns'] = df[price_col].pct_change()
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # Price momentum (key features for direction prediction)
    df['momentum_3'] = df[price_col].pct_change(3)
    df['momentum_5'] = df[price_col].pct_change(5)
    df['momentum_10'] = df[price_col].pct_change(10)
    df['momentum_20'] = df[price_col].pct_change(20)
    
    # Rate of Change
    df['roc_5'] = df[price_col].pct_change(5) * 100
    df['roc_10'] = df[price_col].pct_change(10) * 100
    
    # Bollinger Bands
    df['bb_middle'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # Price position indicators
    df['high_5'] = df[price_col].rolling(window=5).max()
    df['low_5'] = df[price_col].rolling(window=5).min()
    df['high_20'] = df[price_col].rolling(window=20).max()
    df['low_20'] = df[price_col].rolling(window=20).min()
    
    df['near_high_5'] = (df[price_col] / df['high_5'] > 0.98).astype(int)
    df['near_low_5'] = (df[price_col] / df['low_5'] < 1.02).astype(int)
    
    # Volume-like indicators (using price action)
    df['price_range'] = (df['high_5'] - df['low_5']) / df[price_col]
    df['price_change_abs'] = abs(df['returns'])
    
    return df

def create_directional_labels(df, price_col='close', horizon=1, threshold=0.0001):
    """
    Create directional labels optimized for XGBoost classification
    """
    print(f"  🎯 Creating directional labels for XGBoost...")
    
    # Calculate future returns
    future_prices = df[price_col].shift(-horizon)
    returns = (future_prices - df[price_col]) / df[price_col]
    
    # Create binary labels (UP/DOWN) for better XGBoost performance
    # XGBoost often performs better with binary classification
    labels = (returns > threshold).astype(int)  # 1 = UP, 0 = DOWN
    
    # Also create magnitude labels for importance
    df['future_return'] = returns
    df['return_magnitude'] = abs(returns)
    
    # Remove last entries
    df = df.iloc[:-horizon].copy()
    df['direction_binary'] = labels[:-horizon]
    
    # Print distribution
    up_count = np.sum(labels[:-horizon] == 1)
    down_count = np.sum(labels[:-horizon] == 0)
    total = len(labels[:-horizon])
    
    print(f"    📊 Binary Label Distribution:")
    print(f"      📈 UP (1): {up_count:,} ({up_count/total*100:.1f}%)")
    print(f"      📉 DOWN (0): {down_count:,} ({down_count/total*100:.1f}%)")
    
    return df

def create_sequences_for_xgboost(data, feature_cols, target_col, n_steps):
    """
    Create feature matrix for XGBoost (flattened sequences)
    """
    X, y = [], []
    
    with tqdm(total=len(data) - n_steps, desc="Creating XGBoost features", unit="samples") as pbar:
        for i in range(n_steps, len(data)):
            # Flatten the sequence into a single feature vector
            sequence_features = []
            
            # Add features from different time steps
            for t in range(n_steps):
                step_idx = i - n_steps + t
                step_features = data[feature_cols].iloc[step_idx].values
                sequence_features.extend(step_features)
            
            # Add current values as additional features
            current_features = data[feature_cols].iloc[i].values
            sequence_features.extend(current_features)
            
            X.append(sequence_features)
            y.append(data[target_col].iloc[i])
            pbar.update(1)
    
    return np.array(X), np.array(y)

# Load data
csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'
print(f"🚀 Loading EURUSD data for XGBoost GPU training...")
print(f"🎮 GPU Status: {'✅ ENABLED' if GPU_AVAILABLE else '❌ CPU ONLY'}")

try:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, sep='\t')
        
        if '<DATE>' in df.columns and '<TIME>' in df.columns and '<CLOSE>' in df.columns:
            print("✅ Detected MetaTrader format")
            
            df['datetime'] = df['<DATE>'] + ' ' + df['<TIME>']
            data = pd.DataFrame()
            data['time'] = df['datetime']
            data['close'] = df['<CLOSE>']
            data['time'] = pd.to_datetime(data['time'], format='%Y.%m.%d %H:%M:%S')
            data = data.set_index('time')
            data['close'] = pd.to_numeric(data['close'], errors='coerce')
            data = data.dropna()
            
            print(f"✅ Loaded {len(data)} records for XGBoost training")
        else:
            raise ValueError("Unrecognized CSV format")
            
except Exception as e:
    print(f"❌ Error: {str(e)}")
    exit(1)

# Feature engineering
print("\n🔧 Step 1/6: XGBoost feature engineering...")
data = add_technical_indicators(data, 'close')

# Create labels
print("\n🎯 Step 2/6: Creating binary directional labels...")
data_with_labels = create_directional_labels(data.copy(), 'close', horizon=1, threshold=0.0001)
data_with_labels = data_with_labels.dropna()

print(f"✅ Final dataset: {len(data_with_labels)} records")

# Feature selection for XGBoost
feature_cols = [col for col in data_with_labels.columns 
                if col not in ['close', 'direction_binary', 'future_return', 'return_magnitude']]
print(f"📊 Using {len(feature_cols)} base features")

# Split data
split_idx = int(len(data_with_labels) * 0.70)
train_data = data_with_labels.iloc[:split_idx]
test_data = data_with_labels.iloc[split_idx:]

print(f"\n📊 Data split:")
print(f"  🏋️ Training: {len(train_data)} records")
print(f"  🧪 Testing: {len(test_data)} records")

# Create features for XGBoost
print("\n📦 Step 3/6: Creating XGBoost feature matrix...")
X_train, y_train = create_sequences_for_xgboost(train_data, feature_cols, 'direction_binary', n_steps)
X_test, y_test = create_sequences_for_xgboost(test_data, feature_cols, 'direction_binary', n_steps)

# Also create price targets
_, y_train_price = create_sequences_for_xgboost(train_data, feature_cols, 'close', n_steps)
_, y_test_price = create_sequences_for_xgboost(test_data, feature_cols, 'close', n_steps)

print(f"✅ XGBoost features: {X_train.shape}")

# Configure XGBoost parameters
if GPU_AVAILABLE:
    direction_params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    price_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'rmse'
    }
else:
    direction_params = {
        'objective': 'binary:logistic',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    
    price_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'rmse',
        'n_jobs': -1
    }

# Train direction classifier
print(f"\n🚀 Step 4/6: Training XGBoost Direction Classifier...")
start_time = time.time()

direction_model = xgb.XGBClassifier(**direction_params)

with tqdm(total=2, desc="XGBoost Training", unit="model") as pbar:
    tqdm.write("  🎯 Training direction classifier...")
    direction_model.fit(X_train, y_train, 
                       eval_set=[(X_test, y_test)], 
                       early_stopping_rounds=50, 
                       verbose=False)
    pbar.update(1)
    
    tqdm.write("  📈 Training price regressor...")
    price_model = xgb.XGBRegressor(**price_params)
    price_model.fit(X_train, y_train_price,
                   eval_set=[(X_test, y_test_price)],
                   early_stopping_rounds=50,
                   verbose=False)
    pbar.update(1)

training_time = time.time() - start_time
print(f"✅ XGBoost training completed in {training_time:.2f} seconds!")

# Evaluate models
print("\n📋 Step 5/6: Evaluating XGBoost performance...")

# Direction predictions
y_pred_direction = direction_model.predict(X_test)
y_pred_direction_proba = direction_model.predict_proba(X_test)

# Price predictions  
y_pred_price = price_model.predict(X_test)

# Calculate metrics
direction_accuracy = accuracy_score(y_test, y_pred_direction)
price_mse = mean_squared_error(y_test_price, y_pred_price)
price_r2 = r2_score(y_test_price, y_pred_price)

print(f"🚀 XGBoost GPU Performance:")
print(f"  🎯 Direction Accuracy: {direction_accuracy:.4f} ({direction_accuracy*100:.2f}%)")
print(f"  📉 Price MSE: {price_mse:.8f}")
print(f"  📈 Price R²: {price_r2:.4f}")
print(f"  ⚡ Training Time: {training_time:.2f} seconds")

# Feature importance analysis
feature_importance = direction_model.feature_importances_
n_base_features = len(feature_cols)

print(f"\n🔍 Top 10 Most Important Features:")
# Create feature names for the flattened features
feature_names = []
for t in range(n_steps):
    for col in feature_cols:
        feature_names.append(f"{col}_t-{n_steps-t}")
# Add current features
for col in feature_cols:
    feature_names.append(f"{col}_current")

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(10)

for i, row in importance_df.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save models
print("\n💾 Step 6/6: Saving XGBoost models...")

os.makedirs('models', exist_ok=True)

# Save XGBoost models
direction_model.save_model('models/xgboost_direction_classifier.json')
price_model.save_model('models/xgboost_price_regressor.json')

# Save model info
model_info = {
    'model_type': 'XGBoost GPU-Accelerated' if GPU_AVAILABLE else 'XGBoost CPU',
    'n_steps': n_steps,
    'n_features': len(feature_names),
    'training_size': len(X_train),
    'test_size': len(X_test),
    'direction_accuracy': float(direction_accuracy),
    'price_mse': float(price_mse),
    'price_r2': float(price_r2),
    'training_time_seconds': training_time,
    'gpu_enabled': GPU_AVAILABLE,
    'direction_params': direction_params,
    'price_params': price_params,
    'top_features': importance_df['feature'].tolist()
}

with open('models/xgboost_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"✅ XGBoost models saved:")
print(f"  🎯 Direction: models/xgboost_direction_classifier.json")
print(f"  📈 Price: models/xgboost_price_regressor.json") 
print(f"  📋 Info: models/xgboost_model_info.json")

print(f"\n🎉 XGBoost GPU training completed!")
print(f"🚀 Final Performance:")
print(f"  🎯 Direction Accuracy: {direction_accuracy*100:.2f}%")
print(f"  📈 Price R²: {price_r2:.4f}")
print(f"  ⚡ Speed: {training_time:.2f}s")
print(f"  🎮 GPU: {'✅ ENABLED' if GPU_AVAILABLE else '❌ CPU ONLY'}")

if not GPU_AVAILABLE:
    print(f"\n💡 To enable XGBoost GPU:")
    print(f"  pip install xgboost")
    print(f"  # Requires NVIDIA GPU with CUDA or AMD GPU with ROCm") 