# Random Forest + XGBoost Forex Prediction

A streamlined machine learning framework for forex price prediction using hybrid CPU/GPU architecture with Random Forest for direction classification and XGBoost for price regression.

## 🎯 Project Overview

**Hybrid Architecture**: Random Forest (CPU) for market direction + XGBoost (GPU) for price prediction  
**Focus**: Binary classification (UP/DOWN) with enhanced technical indicators  
**Target**: EURUSD H1 timeframe with MetaTrader 5 integration

## ⚡ Key Features

- **🌲 Random Forest Classifier (CPU)**: Binary direction prediction (UP/DOWN only)
- **🚀 XGBoost Regressor (GPU)**: Precise price prediction with CUDA acceleration
- **📊 40+ Technical Indicators**: Enhanced feature engineering for better accuracy
- **🎯 Binary Classification**: Focused on clear directional movements, filtering sideways action
- **📈 MetaTrader 5 Ready**: ONNX export for live trading integration
- **⚡ GPU Acceleration**: XGBoost GPU support with CPU fallback

## 📁 Project Structure

```
random-forest-xgboost/
├── 🏆 Main Models
│   ├── price_direction_random_forest_xgboost_model.py  # MAIN HYBRID MODEL
│
├── 🧪 Testing & Validation  
│   ├── test_model.py                             # Model testing suite
│   ├── test_gpu.py                               # GPU capability testing
│   └── merge_eurusd_data.py                      # Data preprocessing
│
├── 📈 MetaTrader Integration
│   └── random_forest_xgboost.mq5                 # Expert Advisor for MT5
│
├── 💾 Model Storage
│   └── models/                                   # Saved models and metadata
│       ├── direction_classifier.pkl              # Random Forest classifier
│       ├── price_regressor.pkl                   # XGBoost regressor
│       ├── improved_model_info.json              # Model metadata
│       └── random_forest_xgboost.onnx            # ONNX export
│
└── 📋 Configuration
    ├── requirements.txt                          # Full dependencies
    └── README.md                                 # This file
```

## 🏆 Main Model: `price_direction_random_forest_xgboost_model.py`

### Hybrid Architecture

**Direction Classifier (Random Forest - CPU)**
```
Pipeline:
├── StandardScaler preprocessing
└── RandomForestClassifier
    ├── 100 estimators
    ├── Max depth: 15
    ├── Balanced class weights
    └── CPU optimization (n_jobs=-1)
```

**Price Regressor (XGBoost - GPU)**
```
Pipeline:
├── StandardScaler preprocessing
└── XGBRegressor
    ├── 100 estimators  
    ├── GPU acceleration (CUDA)
    ├── Histogram tree method
    └── Optimized hyperparameters
```

### Technical Indicators (40+ Features)

- **Moving Averages**: SMA (5,10,20,50), EMA (5,10,20)
- **Price Ratios**: Price vs moving averages, crossover signals
- **Momentum**: ROC, momentum indicators, rate of change
- **Volatility**: Rolling standard deviation, Bollinger Bands
- **Technical**: RSI, Bollinger Band position, price extremes
- **Pattern Recognition**: High/low distances, trend indicators

### Data Processing Pipeline

1. **📈 Data Loading**: EURUSD H1 CSV from MetaTrader
2. **🔧 Feature Engineering**: Generate 40+ technical indicators
3. **🎯 Binary Labeling**: Create UP/DOWN labels, filter sideways movements
4. **📦 Sequence Creation**: 100-step time series sequences
5. **🌲 Direction Training**: Train Random Forest classifier (CPU)
6. **🚀 Price Training**: Train XGBoost regressor (GPU)
7. **📊 Evaluation**: Performance metrics and visualization
8. **💾 Export**: Save models and ONNX for MetaTrader

## 🚀 Quick Start

### 1. Install Dependencies

**Full Installation (recommended)**:
```bash
pip install -r requirements.txt
```

**Minimal Installation**:
```bash
pip install -r requirements_basic.txt
```

### 2. Prepare Data
Ensure you have the EURUSD data file:
```
EURUSDm_H1_201801020600_202412310000.csv
```

### 3. Train the Model
```bash
python price_direction_random_forest_xgboost_model.py
```

### 4. Test the Model
```bash
python test_model.py
```

### 5. Check GPU (Optional)
```bash
python test_gpu.py
```

## 📊 Model Performance

### Expected Results

| Component | Metric | Typical Range | GPU Support |
|-----------|--------|---------------|-------------|
| Direction Classifier | Binary Accuracy | 65-75% | ❌ CPU Only |
| Price Regressor | R² Score | 0.80-0.90 | ✅ GPU Accelerated |
| Training Time | Total | 30-60 seconds | Partial GPU |

### Performance Visualization

The model automatically generates `price_direction_random_forest_xgboost_model_performance.png` with:
- Confusion matrix for direction prediction
- Rolling accuracy over time
- Price prediction vs actual comparison
- Feature importance analysis

## ⚙️ GPU Setup

### XGBoost GPU Requirements

1. **CUDA Toolkit**: Version 11.8+ recommended
2. **XGBoost**: Automatically detects GPU, falls back to CPU
3. **Memory**: 4GB+ GPU memory recommended

### GPU Installation

```bash
# Install XGBoost (GPU support included)
pip install xgboost

# For PyTorch (testing only)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### GPU Status Check

```bash
python test_gpu.py
```

## 🔧 Usage Examples

### Basic Training
```python
# Train the hybrid model
python price_direction_random_forest_xgboost.py
```

### Model Loading and Prediction
```python
import joblib
import numpy as np

# Load trained models
direction_model = joblib.load('models/direction_classifier.pkl')
price_model = joblib.load('models/price_regressor.pkl')

# Make predictions
sample_features = np.random.random((1, 4000))  # 100 steps × 40 features
direction_pred = direction_model.predict(sample_features)[0]  # 0=DOWN, 1=UP
price_pred = price_model.predict(sample_features)[0]

print(f"Direction: {'UP' if direction_pred == 1 else 'DOWN'}")
print(f"Price: {price_pred:.5f}")
```

### Custom Feature Engineering
```python
from price_direction_random_forest_xgboost import add_technical_indicators

# Add indicators to your data
enhanced_data = add_technical_indicators(your_dataframe, 'close')
```

## 📈 MetaTrader 5 Integration

### Setup Steps

1. **Copy Expert Advisor**:
   ```
   random_forest_xgboost.mq5 → MQL5/Experts/
   ```

2. **Copy Model Files**:
   ```
   models/*.onnx → MQL5/Files/Models/
   ```

3. **Configure EA**:
   - Timeframe: H1
   - Symbol: EURUSD
   - Model path: Models/random_forest_xgboost.onnx

## 🔍 Model Architecture Details

### Binary Classification Focus

- **Labels**: Only UP (1) and DOWN (0) - no sideways movements
- **Threshold**: 0.0001 (1 pip) minimum movement for labeling
- **Filtering**: Removes ambiguous sideways movements for cleaner signals
- **Balance**: Uses class weights to handle any imbalance

### Sequence Processing

- **Window Size**: 100 time steps (100 hours of H1 data)
- **Features per Step**: 40+ technical indicators
- **Input Shape**: (samples, 4000) - flattened sequences
- **Target**: Direction (binary) + Price (continuous)

## 📋 Dependencies

### Core Requirements
- **pandas**: Data processing
- **numpy**: Numerical computations  
- **scikit-learn**: Random Forest and preprocessing
- **xgboost**: GPU-accelerated gradient boosting
- **joblib**: Model serialization

### Visualization & Export
- **matplotlib/seaborn**: Performance charts
- **onnx/onnxruntime**: MetaTrader export
- **skl2onnx**: Model conversion

### Optional
- **MetaTrader5**: Live data connection
- **torch**: GPU testing utilities

## 🛠️ Troubleshooting

### Common Issues

**GPU Not Detected**:
```bash
# Check CUDA installation
python test_gpu.py

# Install CUDA toolkit if needed
# Download from: https://developer.nvidia.com/cuda-downloads
```

**Memory Issues**:
```python
# Reduce sequence length in model file
n_steps = 50  # Instead of 100
```

**Performance Issues**:
```bash
# Use minimal installation
pip install -r requirements_basic.txt
```

## 🚀 Future Enhancements

- [ ] Multi-currency support (GBPUSD, USDJPY, etc.)
- [ ] Real-time prediction API
- [ ] Advanced ensemble methods
- [ ] Hyperparameter optimization
- [ ] Model interpretability tools

## 🛡️ Disclaimer

This software is for educational and research purposes. Trading involves significant financial risk. Always test thoroughly in demo environments before live deployment.

## 📝 License

Educational and research use. Machine learning implementation for forex prediction. 