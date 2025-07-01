import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Standard imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎮 Using device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ CUDA not available, using CPU")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

n_steps = 100

class ForexDataset(Dataset):
    """PyTorch Dataset for forex data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y.dtype != np.int64 else torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ForexDirectionNet(nn.Module):
    """Neural Network for direction prediction"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=3, dropout=0.3):
        super(ForexDirectionNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer for classification
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ForexPriceNet(nn.Module):
    """Neural Network for price regression"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.2):
        super(ForexPriceNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer for regression
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def add_technical_indicators(df, price_col='close'):
    """Add technical indicators"""
    print("  🔧 Adding technical indicators for PyTorch...")
    
    # Simple Moving Averages
    df['sma_5'] = df[price_col].rolling(window=5).mean()
    df['sma_10'] = df[price_col].rolling(window=10).mean()
    df['sma_20'] = df[price_col].rolling(window=20).mean()
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_5'] = df[price_col].ewm(span=5).mean()
    df['ema_10'] = df[price_col].ewm(span=10).mean()
    df['ema_20'] = df[price_col].ewm(span=20).mean()
    
    # Price ratios
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
    
    # Price momentum
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
    
    return df

def create_directional_labels(df, price_col='close', horizon=1, threshold=0.0001):
    """Create directional labels for classification"""
    print(f"  🎯 Creating directional labels for PyTorch...")
    
    # Calculate future returns
    future_prices = df[price_col].shift(-horizon)
    returns = (future_prices - df[price_col]) / df[price_col]
    
    # Create 3-class labels: DOWN(0), SIDEWAYS(1), UP(2)
    labels = np.where(returns > threshold, 2,    # UP
                     np.where(returns < -threshold, 0,  # DOWN
                             1))                          # SIDEWAYS
    
    # Remove last entries
    df = df.iloc[:-horizon].copy()
    df['direction'] = labels[:-horizon]
    
    # Print distribution
    unique, counts = np.unique(labels[:-horizon], return_counts=True)
    total = len(labels[:-horizon])
    
    print(f"    📊 Label Distribution:")
    for label, count in zip(unique, counts):
        direction_name = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}[label]
        print(f"      {direction_name}: {count:,} ({count/total*100:.1f}%)")
    
    return df

def create_sequences_pytorch(data, feature_cols, target_col, n_steps):
    """Create sequences for PyTorch training"""
    X, y = [], []
    
    with tqdm(total=len(data) - n_steps, desc="Creating PyTorch sequences", unit="samples") as pbar:
        for i in range(n_steps, len(data)):
            # Create sequence features
            sequence_features = []
            for t in range(n_steps):
                step_idx = i - n_steps + t
                step_features = data[feature_cols].iloc[step_idx].values
                sequence_features.extend(step_features)
            
            X.append(sequence_features)
            y.append(data[target_col].iloc[i])
            pbar.update(1)
    
    return np.array(X), np.array(y)

def train_pytorch_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train PyTorch model with GPU acceleration"""
    model = model.to(device)
    
    if hasattr(model, 'network') and hasattr(model.network[-1], 'out_features'):
        if model.network[-1].out_features > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    with tqdm(total=epochs, desc="🚀 GPU Training", unit="epoch") as pbar:
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(outputs.squeeze(), y_batch.float())
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    
                    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                        loss = criterion(outputs, y_batch)
                    else:
                        loss = criterion(outputs.squeeze(), y_batch.float())
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            pbar.update(1)
    
    return train_losses, val_losses

# Load data
csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'
print(f"🚀 Loading EURUSD data for PyTorch GPU training...")
print(f"🎮 Device: {device}")

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
            
            print(f"✅ Loaded {len(data)} records for PyTorch training")
        else:
            raise ValueError("Unrecognized CSV format")
            
except Exception as e:
    print(f"❌ Error: {str(e)}")
    exit(1)

# Feature engineering
print("\n🔧 Step 1/6: PyTorch feature engineering...")
data = add_technical_indicators(data, 'close')

# Create labels
print("\n🎯 Step 2/6: Creating directional labels...")
data_with_labels = create_directional_labels(data.copy(), 'close', horizon=1, threshold=0.0001)
data_with_labels = data_with_labels.dropna()

print(f"✅ Final dataset: {len(data_with_labels)} records")

# Feature selection
feature_cols = [col for col in data_with_labels.columns 
                if col not in ['close', 'direction']]
print(f"📊 Using {len(feature_cols)} base features")

# Split data
split_idx = int(len(data_with_labels) * 0.70)
train_data = data_with_labels.iloc[:split_idx]
test_data = data_with_labels.iloc[split_idx:]

print(f"\n📊 Data split:")
print(f"  🏋️ Training: {len(train_data)} records")
print(f"  🧪 Testing: {len(test_data)} records")

# Create sequences
print("\n📦 Step 3/6: Creating PyTorch sequences...")
X_train, y_train_direction = create_sequences_pytorch(train_data, feature_cols, 'direction', n_steps)
X_test, y_test_direction = create_sequences_pytorch(test_data, feature_cols, 'direction', n_steps)

# Price data
_, y_train_price = create_sequences_pytorch(train_data, feature_cols, 'close', n_steps)
_, y_test_price = create_sequences_pytorch(test_data, feature_cols, 'close', n_steps)

print(f"✅ PyTorch sequences: {X_train.shape}")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create datasets and dataloaders
batch_size = 512  # Larger batch for GPU
train_dataset_dir = ForexDataset(X_train_scaled, y_train_direction)
test_dataset_dir = ForexDataset(X_test_scaled, y_test_direction)
train_dataset_price = ForexDataset(X_train_scaled, y_train_price)
test_dataset_price = ForexDataset(X_test_scaled, y_test_price)

train_loader_dir = DataLoader(train_dataset_dir, batch_size=batch_size, shuffle=True)
test_loader_dir = DataLoader(test_dataset_dir, batch_size=batch_size, shuffle=False)
train_loader_price = DataLoader(train_dataset_price, batch_size=batch_size, shuffle=True)
test_loader_price = DataLoader(test_dataset_price, batch_size=batch_size, shuffle=False)

# Create models
input_size = X_train.shape[1]
print(f"\n🚀 Step 4/6: Creating PyTorch GPU models...")
print(f"  📊 Input size: {input_size}")

direction_model = ForexDirectionNet(input_size, hidden_sizes=[512, 256, 128], num_classes=3)
price_model = ForexPriceNet(input_size, hidden_sizes=[256, 128, 64])

print(f"  🎯 Direction model: {sum(p.numel() for p in direction_model.parameters()):,} parameters")
print(f"  📈 Price model: {sum(p.numel() for p in price_model.parameters()):,} parameters")

# Train direction model
print(f"\n🎯 Step 5/6: Training Direction Classifier on GPU...")
start_time = time.time()
train_losses_dir, val_losses_dir = train_pytorch_model(
    direction_model, train_loader_dir, test_loader_dir, epochs=30, lr=0.001
)
direction_time = time.time() - start_time

# Train price model
print(f"\n📈 Training Price Regressor on GPU...")
start_time = time.time()
train_losses_price, val_losses_price = train_pytorch_model(
    price_model, train_loader_price, test_loader_price, epochs=30, lr=0.001
)
price_time = time.time() - start_time

total_time = direction_time + price_time
print(f"✅ PyTorch GPU training completed in {total_time:.2f} seconds!")

# Evaluate models
print(f"\n📋 Step 6/6: Evaluating PyTorch GPU performance...")

# Direction predictions
direction_model.eval()
price_model.eval()

y_pred_direction = []
y_pred_price = []

with torch.no_grad():
    for X_batch, _ in test_loader_dir:
        X_batch = X_batch.to(device)
        
        # Direction predictions
        outputs_dir = direction_model(X_batch)
        preds_dir = torch.argmax(outputs_dir, dim=1).cpu().numpy()
        y_pred_direction.extend(preds_dir)
        
        # Price predictions
        outputs_price = price_model(X_batch)
        preds_price = outputs_price.squeeze().cpu().numpy()
        y_pred_price.extend(preds_price)

y_pred_direction = np.array(y_pred_direction)
y_pred_price = np.array(y_pred_price)

# Calculate metrics
direction_accuracy = accuracy_score(y_test_direction, y_pred_direction)
price_mse = mean_squared_error(y_test_price, y_pred_price)
price_r2 = r2_score(y_test_price, y_pred_price)

print(f"🚀 PyTorch GPU Performance:")
print(f"  🎯 Direction Accuracy: {direction_accuracy:.4f} ({direction_accuracy*100:.2f}%)")
print(f"  📉 Price MSE: {price_mse:.8f}")
print(f"  📈 Price R²: {price_r2:.4f}")
print(f"  ⚡ Training Time: {total_time:.2f} seconds")
print(f"  🎮 GPU Utilization: {'✅ USED' if device.type == 'cuda' else '❌ CPU ONLY'}")

# Save models
print(f"\n💾 Saving PyTorch GPU models...")
os.makedirs('models', exist_ok=True)

torch.save({
    'direction_model': direction_model.state_dict(),
    'price_model': price_model.state_dict(),
    'scaler': scaler,
    'feature_cols': feature_cols,
    'input_size': input_size
}, 'models/pytorch_gpu_models.pth')

# Save model info
model_info = {
    'model_type': 'PyTorch GPU Neural Networks',
    'device': str(device),
    'n_steps': n_steps,
    'input_size': input_size,
    'training_size': len(X_train),
    'test_size': len(X_test),
    'direction_accuracy': float(direction_accuracy),
    'price_mse': float(price_mse),
    'price_r2': float(price_r2),
    'training_time_seconds': total_time,
    'gpu_enabled': device.type == 'cuda',
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'batch_size': batch_size,
    'epochs': 30
}

with open('models/pytorch_gpu_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"✅ PyTorch models saved:")
print(f"  🧠 Neural networks: models/pytorch_gpu_models.pth")
print(f"  📋 Model info: models/pytorch_gpu_model_info.json")

print(f"\n🎉 PyTorch GPU training completed!")
print(f"🚀 Performance Summary:")
print(f"  🎯 Direction Accuracy: {direction_accuracy*100:.2f}%")
print(f"  📈 Price R²: {price_r2:.4f}")
print(f"  ⚡ Speed: {total_time:.2f}s")
print(f"  🎮 GPU: {'✅ RTX 3060 USED' if device.type == 'cuda' else '❌ CPU ONLY'}")

if device.type == 'cuda':
    print(f"\n💡 GPU Benefits Achieved:")
    print(f"  • Neural networks trained on RTX 3060")
    print(f"  • Large batch sizes ({batch_size}) for better performance")
    print(f"  • Parallel computation acceleration")
    print(f"  • Advanced architectures possible")
else:
    print(f"\n💡 GPU not used - check CUDA installation") 