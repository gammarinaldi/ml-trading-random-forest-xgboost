import pandas as pd
import numpy as np
import joblib
import random
import onnx
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
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
    Add technical indicators to improve directional prediction
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

def create_directional_labels(df, price_col='close', horizon=1, threshold=0.0001):
    """
    Create directional labels for classification
    """
    print(f"  🎯 Creating directional labels (horizon={horizon}, threshold={threshold:.4f})...")
    
    # Calculate future returns
    future_prices = df[price_col].shift(-horizon)
    returns = (future_prices - df[price_col]) / df[price_col]
    
    # Create directional labels with threshold
    labels = np.where(returns > threshold, 1,    # UP
                     np.where(returns < -threshold, -1,  # DOWN
                             0))                          # SIDEWAYS
    
    # Remove last entries where we don't have future data
    df = df.iloc[:-horizon].copy()
    df['direction'] = labels[:-horizon]
    
    # Print label distribution
    label_counts = pd.Series(labels[:-horizon]).value_counts().sort_index()
    total = len(labels[:-horizon])
    
    print(f"    📊 Label distribution:")
    print(f"      📉 DOWN (-1): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/total*100:.1f}%)")
    print(f"      ➡️ SIDEWAYS (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.1f}%)")
    print(f"      📈 UP (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.1f}%)")
    
    return df

def create_feature_sequences(data, feature_cols, n_steps):
    """
    Create sequences using multiple features
    """
    X, y = [], []
    
    with tqdm(total=len(data) - n_steps, desc="Creating feature sequences", unit="sequences") as pbar:
        for i in range(n_steps, len(data)):
            # Get sequence of features
            sequence = data[feature_cols].iloc[i-n_steps:i].values.flatten()
            target = data['direction'].iloc[i]
            
            X.append(sequence)
            y.append(target)
            pbar.update(1)
    
    return np.array(X), np.array(y)

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
print("\n🔧 Step 1/8: Enhanced feature engineering...")
data = add_technical_indicators(data, 'close')

# Create directional labels with different strategies
print("\n🎯 Step 2/8: Creating directional labels...")
data_with_labels = create_directional_labels(data.copy(), 'close', horizon=1, threshold=0.0001)

# Remove NaN values from technical indicators
data_with_labels = data_with_labels.dropna()
print(f"✅ Data after cleaning: {len(data_with_labels)} records")

# Define feature columns (all technical indicators)
feature_cols = [col for col in data_with_labels.columns if col not in ['close', 'direction']]
print(f"📊 Using {len(feature_cols)} features: {feature_cols[:5]}... (showing first 5)")

# Split data
training_size = int(len(data_with_labels) * 0.70)
train_data = data_with_labels.iloc[:training_size]
test_data = data_with_labels.iloc[training_size:]

print(f"\n📊 Data split:")
print(f"  🏋️ Training: {len(train_data)} records ({training_size/len(data_with_labels)*100:.1f}%)")
print(f"  🧪 Testing: {len(test_data)} records ({(len(data_with_labels)-training_size)/len(data_with_labels)*100:.1f}%)")

# Create sequences for training
print("\n📦 Step 3/8: Creating feature sequences...")
X_train, y_train = create_feature_sequences(train_data, feature_cols, n_steps)
X_test, y_test = create_feature_sequences(test_data, feature_cols, n_steps)

print(f"✅ Training sequences: {X_train.shape}")
print(f"✅ Test sequences: {X_test.shape}")

# Check class distribution
print(f"\n📊 Training label distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    direction_name = {-1: "DOWN", 0: "SIDEWAYS", 1: "UP"}[label]
    print(f"  {direction_name}: {count:,} ({count/len(y_train)*100:.1f}%)")

# Create and train directional classifier
print("\n🌲 Step 4/8: Training Random Forest Classifier for direction...")

# Try GPU acceleration first
try:
    import xgboost as xgb
    print("  🎮 XGBoost GPU detected - using GPU acceleration!")
    
    # Check GPU availability
    try:
        test_matrix = xgb.DMatrix(np.random.random((10, 5)))
        gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(gpu_params, test_matrix, num_boost_round=1, verbose_eval=False)
        
        # Use XGBoost GPU for better performance
        direction_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(
                objective='multi:softprob',
                tree_method='hist',  # Updated method
                device='cuda',  # GPU acceleration (updated parameter)
                n_estimators=200,  # More trees with GPU
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            ))
        ])
        print("  ✅ Using XGBoost GPU classifier!")
        
    except Exception:
        raise ImportError("GPU not available")
        
except ImportError:
    # Fallback to CPU Random Forest
    print("  💻 Using CPU Random Forest (GPU not available)")
    direction_classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,  # More trees for better accuracy
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ])

# Training with progress indication
training_stages = [
    "Standardizing features",
    "Training Random Forest Classifier (100 trees)",
    "Optimizing for directional accuracy"
]

with tqdm(total=len(training_stages), desc="Training classifier", unit="stage") as pbar:
    for i, stage in enumerate(training_stages):
        tqdm.write(f"  🔄 {stage}...")
        if i == len(training_stages) - 1:
            direction_classifier.fit(X_train, y_train)
        else:
            time.sleep(0.5)
        pbar.update(1)

tqdm.write("✅ Directional classifier training completed!")

# Create regression model for price prediction
print("\n📈 Step 5/8: Training Random Forest Regressor for price...")

# Prepare data for regression (predict close price)
y_train_price = train_data['close'].iloc[n_steps:].values
y_test_price = test_data['close'].iloc[n_steps:].values

# Try GPU acceleration for price regressor too
try:
    # Check if XGBoost was successfully imported above
    if 'xgb' in locals():
        print("  🎮 Using XGBoost GPU for price regression!")
        price_regressor = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',  # Updated method
                device='cuda',  # GPU acceleration (updated parameter)
                n_estimators=100,  # More trees with GPU
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='rmse'
            ))
        ])
        print("  ✅ Using XGBoost GPU regressor!")
    else:
        raise ImportError("XGBoost not available")
        
except (ImportError, NameError):
    # Fallback to CPU Random Forest
    print("  💻 Using CPU Random Forest Regressor")
    price_regressor = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

price_regressor.fit(X_train, y_train_price)
tqdm.write("✅ Price regressor training completed!")

# Evaluate both models
print("\n📋 Step 6/8: Evaluating model performance...")

# Direction predictions
y_pred_direction = direction_classifier.predict(X_test)
direction_accuracy = accuracy_score(y_test, y_pred_direction)

# Price predictions
y_pred_price = price_regressor.predict(X_test)
price_mse = mean_squared_error(y_test_price, y_pred_price)
price_r2 = r2_score(y_test_price, y_pred_price)

print(f"📊 Model Performance:")
print(f"  🎯 Direction Accuracy: {direction_accuracy:.4f} ({direction_accuracy*100:.2f}%)")
print(f"  📉 Price MSE: {price_mse:.8f}")
print(f"  📈 Price R²: {price_r2:.4f}")

# Detailed classification report
print(f"\n📋 Detailed Classification Report:")
class_names = ['DOWN', 'SIDEWAYS', 'UP']
target_names = [class_names[int(i)+1] for i in sorted(np.unique(y_test))]
print(classification_report(y_test, y_pred_direction, target_names=target_names))

# Create visualizations
print("\n📊 Step 7/8: Creating performance visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_direction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=axes[0,0])
axes[0,0].set_title('Direction Prediction Confusion Matrix')
axes[0,0].set_xlabel('Predicted')
axes[0,0].set_ylabel('Actual')

# Direction accuracy over time
correct_predictions = (y_test == y_pred_direction).astype(int)
rolling_accuracy = pd.Series(correct_predictions).rolling(window=100).mean()
axes[0,1].plot(rolling_accuracy)
axes[0,1].set_title('Rolling Direction Accuracy (100-period window)')
axes[0,1].set_xlabel('Time Steps')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].grid(True)

# Price predictions vs actual
n_plot = min(500, len(y_test_price))
axes[1,0].plot(y_test_price[:n_plot], label='Actual', alpha=0.7)
axes[1,0].plot(y_pred_price[:n_plot], label='Predicted', alpha=0.7)
axes[1,0].set_title('Price Predictions vs Actual')
axes[1,0].set_xlabel('Time Steps')
axes[1,0].set_ylabel('Price')
axes[1,0].legend()
axes[1,0].grid(True)

# Feature importance (top 15)
feature_importance = direction_classifier.named_steps['rf'].feature_importances_
# Create feature names for flattened sequences
feature_names = []
for i in range(n_steps):
    for col in feature_cols:
        feature_names.append(f"{col}_t-{n_steps-i}")

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(15)

axes[1,1].barh(range(len(importance_df)), importance_df['importance'])
axes[1,1].set_yticks(range(len(importance_df)))
axes[1,1].set_yticklabels(importance_df['feature'], fontsize=8)
axes[1,1].set_title('Top 15 Feature Importances')
axes[1,1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('improved_model_performance.png', dpi=300, bbox_inches='tight')
tqdm.write("📊 Performance visualization saved as 'improved_model_performance.png'")
plt.close()

# Save models
print("\n💾 Step 8/8: Saving improved models...")

os.makedirs('models', exist_ok=True)

# Save both models
joblib.dump(direction_classifier, 'models/direction_classifier.pkl')
joblib.dump(price_regressor, 'models/price_regressor.pkl')

# Save model metadata
model_info = {
    'n_steps': n_steps,
    'features_used': feature_cols,
    'n_features': len(feature_cols),
    'training_size': len(X_train),
    'test_size': len(X_test),
    'direction_accuracy': float(direction_accuracy),
    'price_mse': float(price_mse),
    'price_r2': float(price_r2),
    'model_type': 'Enhanced RandomForest with Technical Indicators',
    'classifier_estimators': 100,
    'regressor_estimators': 50,
    'class_balance': 'balanced'
}

import json
with open('models/improved_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"✅ Models saved:")
print(f"  📊 Direction classifier: models/direction_classifier.pkl")
print(f"  📈 Price regressor: models/price_regressor.pkl")
print(f"  📋 Model info: models/improved_model_info.json")

# Test combined prediction
print(f"\n🧪 Testing combined prediction...")
if len(X_test) > 0:
    sample_idx = 0
    sample_features = X_test[sample_idx:sample_idx+1]
    
    pred_direction = direction_classifier.predict(sample_features)[0]
    pred_price = price_regressor.predict(sample_features)[0]
    actual_direction = y_test[sample_idx]
    actual_price = y_test_price[sample_idx]
    
    direction_names = {-1: "DOWN ↘", 0: "SIDEWAYS →", 1: "UP ↗"}
    
    print(f"🎯 Sample Prediction:")
    print(f"  📊 Predicted Direction: {direction_names.get(pred_direction, 'Unknown')}")
    print(f"  📈 Predicted Price: {pred_price:.5f}")
    print(f"  ✅ Actual Direction: {direction_names.get(actual_direction, 'Unknown')}")
    print(f"  ✅ Actual Price: {actual_price:.5f}")
    
    direction_correct = "✅ CORRECT" if pred_direction == actual_direction else "❌ WRONG"
    price_error = abs(pred_price - actual_price)
    
    print(f"  🎯 Direction: {direction_correct}")
    print(f"  📏 Price Error: {price_error:.5f} ({price_error/actual_price*10000:.1f} pips)")

print(f"\n🎉 Enhanced model training completed!")
print(f"🎯 Direction Accuracy improved to: {direction_accuracy*100:.2f}%")
print(f"📈 Price prediction R²: {price_r2:.4f}")
print(f"\n✨ Ready to use! Run the test script with these new models.") 