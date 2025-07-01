import pandas as pd
import numpy as np
import joblib
import random
import os
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRFC
    from cuml.ensemble import RandomForestRegressor as cuRFR
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.metrics import accuracy_score as cu_accuracy_score
    from cuml.model_selection import train_test_split as cu_train_test_split
    print("✅ RAPIDS cuML imported successfully - GPU acceleration enabled!")
    GPU_AVAILABLE = True
except ImportError:
    print("❌ RAPIDS cuML not available. Install with: conda install -c rapidsai -c conda-forge cuml")
    print("🔄 Falling back to scikit-learn CPU version...")
    from sklearn.ensemble import RandomForestClassifier as cuRFC
    from sklearn.ensemble import RandomForestRegressor as cuRFR
    from sklearn.preprocessing import StandardScaler as cuStandardScaler
    from sklearn.metrics import accuracy_score as cu_accuracy_score
    from sklearn.model_selection import train_test_split as cu_train_test_split
    GPU_AVAILABLE = False

# Standard imports
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
    Add technical indicators to improve directional prediction
    """
    print("  🔧 Adding technical indicators...")
    
    # Convert to cudf if GPU available
    if GPU_AVAILABLE:
        if not isinstance(df, cudf.DataFrame):
            df = cudf.from_pandas(df)
    
    # Simple Moving Averages
    df['sma_5'] = df[price_col].rolling(window=5).mean()
    df['sma_10'] = df[price_col].rolling(window=10).mean()
    df['sma_20'] = df[price_col].rolling(window=20).mean()
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    
    # Exponential Moving Averages (approximation for cudf)
    alpha_5 = 2.0 / (5 + 1)
    alpha_10 = 2.0 / (10 + 1)
    alpha_20 = 2.0 / (20 + 1)
    
    df['ema_5'] = df[price_col].ewm(alpha=alpha_5).mean()
    df['ema_10'] = df[price_col].ewm(alpha=alpha_10).mean()
    df['ema_20'] = df[price_col].ewm(alpha=alpha_20).mean()
    
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
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
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
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
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

def create_directional_labels(df, price_col='close', horizon=1, threshold=0.0001):
    """
    Create directional labels for classification
    """
    print(f"  🎯 Creating directional labels (horizon={horizon}, threshold={threshold:.4f})...")
    
    # Convert to pandas if needed for label creation
    if GPU_AVAILABLE and isinstance(df, cudf.DataFrame):
        df_pandas = df.to_pandas()
    else:
        df_pandas = df
    
    # Calculate future returns
    future_prices = df_pandas[price_col].shift(-horizon)
    returns = (future_prices - df_pandas[price_col]) / df_pandas[price_col]
    
    # Create directional labels with threshold
    labels = np.where(returns > threshold, 1,    # UP
                     np.where(returns < -threshold, -1,  # DOWN
                             0))                          # SIDEWAYS
    
    # Remove last entries where we don't have future data
    df_result = df_pandas.iloc[:-horizon].copy()
    df_result['direction'] = labels[:-horizon]
    
    # Print label distribution
    label_counts = pd.Series(labels[:-horizon]).value_counts().sort_index()
    total = len(labels[:-horizon])
    
    print(f"    📊 Label distribution:")
    print(f"      📉 DOWN (-1): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/total*100:.1f}%)")
    print(f"      ➡️ SIDEWAYS (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.1f}%)")
    print(f"      📈 UP (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.1f}%)")
    
    # Convert back to cudf if GPU available
    if GPU_AVAILABLE:
        return cudf.from_pandas(df_result)
    else:
        return df_result

def create_feature_sequences_gpu(data, feature_cols, n_steps):
    """
    Create sequences using multiple features with GPU optimization
    """
    # Convert to pandas for sequence creation (more efficient for this operation)
    if GPU_AVAILABLE and isinstance(data, cudf.DataFrame):
        data_pandas = data.to_pandas()
    else:
        data_pandas = data
    
    X, y = [], []
    
    with tqdm(total=len(data_pandas) - n_steps, desc="Creating feature sequences", unit="sequences") as pbar:
        for i in range(n_steps, len(data_pandas)):
            # Get sequence of features
            sequence = data_pandas[feature_cols].iloc[i-n_steps:i].values.flatten()
            target = data_pandas['direction'].iloc[i]
            
            X.append(sequence)
            y.append(target)
            pbar.update(1)
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert to cudf if GPU available
    if GPU_AVAILABLE:
        X = cudf.DataFrame(X)
        y = cudf.Series(y)
    
    return X, y

# Load data from CSV file
csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'

print(f"🚀 Loading EURUSD data for GPU-accelerated training...")
print(f"🎮 GPU Status: {'✅ ENABLED' if GPU_AVAILABLE else '❌ DISABLED (using CPU)'}")

try:
    # Load CSV data
    if os.path.exists(csv_file):
        print(f"Found CSV file: {csv_file}")
        
        # Load with pandas first, then convert to cudf if available
        df = pd.read_csv(csv_file, sep='\t')
        
        print("CSV file structure:")
        print(f"Columns: {list(df.columns)}")
        
        # Check MetaTrader format
        if '<DATE>' in df.columns and '<TIME>' in df.columns and '<CLOSE>' in df.columns:
            print("✅ Detected MetaTrader export format")
            
            # Combine date and time columns
            df['datetime'] = df['<DATE>'] + ' ' + df['<TIME>']
            
            # Create data DataFrame
            data = pd.DataFrame()
            data['time'] = df['datetime']
            data['close'] = df['<CLOSE>']
            
            # Convert time column
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
print("\n🔧 Step 1/8: Enhanced feature engineering with GPU acceleration...")
data = add_technical_indicators(data, 'close')

# Create directional labels
print("\n🎯 Step 2/8: Creating directional labels...")
data_with_labels = create_directional_labels(data.copy(), 'close', horizon=1, threshold=0.0001)

# Remove NaN values
if GPU_AVAILABLE and isinstance(data_with_labels, cudf.DataFrame):
    data_with_labels = data_with_labels.dropna()
else:
    data_with_labels = data_with_labels.dropna()

print(f"✅ Data after cleaning: {len(data_with_labels)} records")

# Define feature columns
feature_cols = [col for col in data_with_labels.columns if col not in ['close', 'direction']]
print(f"📊 Using {len(feature_cols)} features for GPU training")

# Split data
training_size = int(len(data_with_labels) * 0.70)

if GPU_AVAILABLE and isinstance(data_with_labels, cudf.DataFrame):
    train_data = data_with_labels.iloc[:training_size]
    test_data = data_with_labels.iloc[training_size:]
else:
    train_data = data_with_labels.iloc[:training_size]
    test_data = data_with_labels.iloc[training_size:]

print(f"\n📊 Data split:")
print(f"  🏋️ Training: {len(train_data)} records ({training_size/len(data_with_labels)*100:.1f}%)")
print(f"  🧪 Testing: {len(test_data)} records ({(len(data_with_labels)-training_size)/len(data_with_labels)*100:.1f}%)")

# Create sequences for training
print("\n📦 Step 3/8: Creating feature sequences...")
X_train, y_train = create_feature_sequences_gpu(train_data, feature_cols, n_steps)
X_test, y_test = create_feature_sequences_gpu(test_data, feature_cols, n_steps)

print(f"✅ Training sequences: {X_train.shape}")
print(f"✅ Test sequences: {X_test.shape}")

# Create and train GPU-accelerated directional classifier
print(f"\n🚀 Step 4/8: Training GPU-accelerated Random Forest Classifier...")

# Configure GPU Random Forest
if GPU_AVAILABLE:
    print("  🎮 Using RAPIDS cuML GPU Random Forest")
    direction_classifier = cuRFC(
        n_estimators=200,  # More trees for GPU
        max_depth=20,      # Deeper trees
        max_features='sqrt',
        random_state=42,
        bootstrap=True
    )
else:
    print("  💻 Using scikit-learn CPU Random Forest")
    from sklearn.ensemble import RandomForestClassifier
    direction_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

# Training with timing
start_time = time.time()

with tqdm(total=3, desc="GPU Training", unit="stage") as pbar:
    tqdm.write("  🔄 Preparing data for GPU training...")
    
    # Convert data for training
    if GPU_AVAILABLE:
        if isinstance(X_train, cudf.DataFrame):
            X_train_fit = X_train
            y_train_fit = y_train
        else:
            X_train_fit = cudf.DataFrame(X_train)
            y_train_fit = cudf.Series(y_train)
    else:
        X_train_fit = np.array(X_train) if hasattr(X_train, 'values') else X_train
        y_train_fit = np.array(y_train) if hasattr(y_train, 'values') else y_train
    
    pbar.update(1)
    
    tqdm.write("  🚀 Training Random Forest on GPU...")
    direction_classifier.fit(X_train_fit, y_train_fit)
    pbar.update(1)
    
    tqdm.write("  ✅ GPU training completed!")
    pbar.update(1)

training_time = time.time() - start_time
tqdm.write(f"✅ Directional classifier training completed in {training_time:.2f} seconds!")

# Create GPU-accelerated price regressor
print("\n📈 Step 5/8: Training GPU-accelerated Price Regressor...")

# Prepare price data
if GPU_AVAILABLE and isinstance(train_data, cudf.DataFrame):
    y_train_price = train_data['close'].iloc[n_steps:].values
    y_test_price = test_data['close'].iloc[n_steps:].values
else:
    y_train_price = train_data['close'].iloc[n_steps:].values
    y_test_price = test_data['close'].iloc[n_steps:].values

if GPU_AVAILABLE:
    price_regressor = cuRFR(
        n_estimators=100,
        max_depth=15,
        max_features='sqrt',
        random_state=42
    )
    
    if not isinstance(y_train_price, cudf.Series):
        y_train_price = cudf.Series(y_train_price)
else:
    from sklearn.ensemble import RandomForestRegressor
    price_regressor = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

price_regressor.fit(X_train_fit, y_train_price)
tqdm.write("✅ GPU price regressor training completed!")

# Evaluate models
print("\n📋 Step 6/8: Evaluating GPU model performance...")

# Make predictions
if GPU_AVAILABLE:
    if isinstance(X_test, pd.DataFrame):
        X_test_pred = cudf.DataFrame(X_test)
    else:
        X_test_pred = X_test
        
    y_pred_direction = direction_classifier.predict(X_test_pred)
    y_pred_price = price_regressor.predict(X_test_pred)
    
    # Convert predictions back to numpy if needed
    if hasattr(y_pred_direction, 'values'):
        y_pred_direction = y_pred_direction.values
    if hasattr(y_pred_price, 'values'):
        y_pred_price = y_pred_price.values
    if hasattr(y_test, 'values'):
        y_test_eval = y_test.values
    else:
        y_test_eval = y_test
else:
    X_test_pred = np.array(X_test) if hasattr(X_test, 'values') else X_test
    y_pred_direction = direction_classifier.predict(X_test_pred)
    y_pred_price = price_regressor.predict(X_test_pred)
    y_test_eval = np.array(y_test) if hasattr(y_test, 'values') else y_test

# Calculate metrics
from sklearn.metrics import accuracy_score
direction_accuracy = accuracy_score(y_test_eval, y_pred_direction)
price_mse = mean_squared_error(y_test_price, y_pred_price)
price_r2 = r2_score(y_test_price, y_pred_price)

print(f"🚀 GPU-Accelerated Model Performance:")
print(f"  🎯 Direction Accuracy: {direction_accuracy:.4f} ({direction_accuracy*100:.2f}%)")
print(f"  📉 Price MSE: {price_mse:.8f}")
print(f"  📈 Price R²: {price_r2:.4f}")
print(f"  ⚡ Training Time: {training_time:.2f} seconds")

# Save GPU models
print("\n💾 Step 7/8: Saving GPU-trained models...")

os.makedirs('models', exist_ok=True)

# Convert models to CPU format for saving
if GPU_AVAILABLE:
    # For cuML models, we need to save them differently
    import pickle
    
    # Save cuML models with pickle
    with open('models/gpu_direction_classifier.pkl', 'wb') as f:
        pickle.dump(direction_classifier, f)
    
    with open('models/gpu_price_regressor.pkl', 'wb') as f:
        pickle.dump(price_regressor, f)
    
    print("✅ GPU models saved (cuML format)")
else:
    # Save sklearn models normally
    joblib.dump(direction_classifier, 'models/gpu_direction_classifier.pkl')
    joblib.dump(price_regressor, 'models/gpu_price_regressor.pkl')
    print("✅ CPU models saved")

# Save enhanced model info
model_info = {
    'n_steps': n_steps,
    'features_used': feature_cols,
    'n_features': len(feature_cols),
    'training_size': len(X_train),
    'test_size': len(X_test),
    'direction_accuracy': float(direction_accuracy),
    'price_mse': float(price_mse),
    'price_r2': float(price_r2),
    'training_time_seconds': training_time,
    'model_type': 'GPU-Accelerated RandomForest' if GPU_AVAILABLE else 'CPU RandomForest',
    'gpu_enabled': GPU_AVAILABLE,
    'classifier_estimators': 200 if GPU_AVAILABLE else 100,
    'regressor_estimators': 100 if GPU_AVAILABLE else 50
}

with open('models/gpu_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"✅ Models saved:")
print(f"  📊 Direction classifier: models/gpu_direction_classifier.pkl")
print(f"  📈 Price regressor: models/gpu_price_regressor.pkl")
print(f"  📋 Model info: models/gpu_model_info.json")

print(f"\n🎉 GPU-accelerated model training completed!")
print(f"🚀 Performance Summary:")
print(f"  🎯 Direction Accuracy: {direction_accuracy*100:.2f}%")
print(f"  📈 Price R²: {price_r2:.4f}")
print(f"  ⚡ Training Speed: {training_time:.2f}s")
print(f"  🎮 GPU Acceleration: {'✅ ENABLED' if GPU_AVAILABLE else '❌ DISABLED'}")

if GPU_AVAILABLE:
    print(f"\n💡 GPU Benefits:")
    print(f"  • Faster training with more trees (200 vs 100)")
    print(f"  • Deeper trees (20 vs 15 levels)")
    print(f"  • Better memory efficiency")
    print(f"  • Scalable to larger datasets")
else:
    print(f"\n💡 To enable GPU acceleration:")
    print(f"  conda install -c rapidsai -c conda-forge cuml")
    print(f"  # Requires NVIDIA GPU with CUDA support") 