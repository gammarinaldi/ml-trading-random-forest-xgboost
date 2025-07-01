# Random Forest Model for MetaTrader 5

This project implements a Random Forest regression model with advanced preprocessing for forex price prediction, based on the MQL5 article: [Python, ONNX and MetaTrader 5: Creating a RandomForest model with RobustScaler and PolynomialFeatures data preprocessing](https://www.mql5.com/en/articles/13725).

## Features

- **Random Forest Regression**: Uses scikit-learn's RandomForestRegressor with 20 estimators
- **Advanced Preprocessing Pipeline**:
  - RobustScaler for outlier-resistant normalization
  - PolynomialFeatures for feature engineering (degree=2)
  - MinMaxScaler for final normalization
- **ONNX Export**: Model exported to ONNX format for MetaTrader 5 integration
- **Local Storage**: All models and data saved locally
- **MetaTrader 5 Integration**: Complete Expert Advisor for automated trading
- **Comprehensive Testing**: Model validation and performance visualization

## Project Structure

```
random-forest-robusts-scaler/
├── random_forest_model.py      # Main model training script
├── test_model.py              # Model testing and validation script
├── RandomForestEA.mq5         # MetaTrader 5 Expert Advisor
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Generated model files
│   ├── random_forest_pipeline.pkl    # Complete sklearn pipeline
│   ├── random_forest_model.onnx      # ONNX model for MT5
│   └── model_info.json              # Model metadata
├── model_performance.png      # Training performance visualization
└── model_test_visualization.png # Test results visualization
```

## Requirements

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### MetaTrader 5 (Optional)

- MetaTrader 5 platform for live data and trading
- If not available, the script will generate synthetic EURUSD data for demonstration

## Installation and Setup

1. **Clone or download this project**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python random_forest_model.py
   ```

4. **Test the model**:
   ```bash
   python test_model.py
   ```

5. **For MetaTrader 5 integration**:
   - Copy `RandomForestEA.mq5` to your MetaTrader 5 Experts folder
   - Copy `models/random_forest_model.onnx` to your MetaTrader 5 `Files/Models/` folder
   - Compile and run the Expert Advisor

## Usage

### Training the Model

Run the main training script:

```bash
python random_forest_model.py
```

This script will:
- Initialize MetaTrader 5 (or create synthetic data if not available)
- Load EURUSD historical data from 2000-2023
- Preprocess the data using the sklearn pipeline
- Train the Random Forest model
- Export to ONNX format
- Save all models and metadata locally
- Generate performance visualizations

### Testing the Model

Test the trained model:

```bash
python test_model.py
```

This script will:
- Load the saved model pipeline
- Generate sample price data for testing
- Make predictions and calculate accuracy metrics
- Create visualization plots
- Test directional accuracy

### MetaTrader 5 Integration

1. **Copy files to MetaTrader 5**:
   - Copy `RandomForestEA.mq5` to `MQL5/Experts/`
   - Copy `models/random_forest_model.onnx` to `MQL5/Files/Models/`

2. **Compile the Expert Advisor** in MetaTrader 5

3. **Configure parameters**:
   - `Lots`: Position size (default: 0.1)
   - `StopLoss`: Stop loss in points (default: 500)
   - `TakeProfit`: Take profit in points (default: 1000)
   - `Magic`: Magic number for trade identification (default: 123456)
   - `ModelPath`: Path to ONNX model (default: "Models\\random_forest_model.onnx")

4. **Run the Expert Advisor** on EURUSD H1 timeframe

## Model Architecture

### Preprocessing Pipeline

1. **RobustScaler**: Normalizes data using median and IQR, making it robust to outliers
2. **PolynomialFeatures**: Creates polynomial and interaction features (degree=2)
3. **MinMaxScaler**: Final normalization to [0,1] range

### Random Forest Configuration

- **n_estimators**: 20 decision trees
- **max_depth**: 10 levels maximum
- **min_samples_split**: 5 samples minimum to split
- **min_samples_leaf**: 2 samples minimum per leaf
- **Time steps**: 100 previous price points used for prediction

### Model Input/Output

- **Input**: 100 consecutive close prices
- **Output**: Predicted next close price
- **Format**: ONNX for MetaTrader 5 compatibility

## Performance Metrics

The model tracks several performance metrics:

- **MSE** (Mean Squared Error): Average squared difference between predictions and actual values
- **R² Score**: Coefficient of determination (proportion of variance explained)
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **Direction Accuracy**: Percentage of correct directional predictions

## Trading Strategy

The Expert Advisor implements a simple trading strategy:

1. **Signal Generation**: Compare predicted price with current price
2. **Buy Signal**: If predicted price > current price (and spread < 0.5 pips)
3. **Sell Signal**: If predicted price < current price (and spread < 0.5 pips)
4. **Risk Management**: Fixed stop loss and take profit levels
5. **Position Management**: Close opposing positions when opening new ones

## Files Description

### `random_forest_model.py`
Main training script that:
- Loads data from MetaTrader 5 or generates synthetic data
- Creates preprocessing pipeline
- Trains Random Forest model
- Exports to ONNX format
- Saves all models locally

### `test_model.py`
Model testing script that:
- Loads saved models
- Tests predictions on sample data
- Calculates performance metrics
- Creates visualization plots

### `RandomForestEA.mq5`
MetaTrader 5 Expert Advisor that:
- Loads ONNX model
- Makes real-time predictions
- Executes trades based on predictions
- Manages positions and risk

### `requirements.txt`
Python package dependencies with minimum versions

## Customization

### Model Parameters

You can modify model parameters in `random_forest_model.py`:

```python
# Time steps for input sequence
n_steps = 100

# Random Forest parameters
RandomForestRegressor(
    n_estimators=20,      # Number of trees
    max_depth=10,         # Maximum tree depth
    min_samples_split=5,  # Minimum samples to split
    min_samples_leaf=2,   # Minimum samples per leaf
    random_state=42,
    n_jobs=-1
)

# Polynomial features degree
PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
```

### Trading Parameters

Modify trading parameters in the MetaTrader 5 Expert Advisor:

```mql5
input double   Lots = 0.1;           // Position size
input int      StopLoss = 500;       // Stop loss in points
input int      TakeProfit = 1000;    // Take profit in points
input int      Magic = 123456;       // Magic number
```

## Troubleshooting

### Common Issues

1. **MetaTrader 5 not found**: The script will automatically generate synthetic data for demonstration

2. **ONNX conversion fails**: This may happen due to compatibility issues with PolynomialFeatures. The sklearn pipeline will still be saved.

3. **Model files not found**: Ensure you run `random_forest_model.py` before `test_model.py`

4. **MetaTrader 5 ONNX model not loading**: 
   - Check that the ONNX file is in the correct folder: `MQL5/Files/Models/`
   - Verify the file path in the Expert Advisor parameters
   - Ensure MetaTrader 5 has ONNX support enabled

### Performance Considerations

- **Training time**: Depends on data size and system resources
- **Memory usage**: PolynomialFeatures significantly increases feature count
- **Prediction speed**: Real-time predictions are fast enough for forex trading

## Limitations

1. **Model complexity**: Simple 20-tree Random Forest may not capture complex market patterns
2. **Feature engineering**: Limited to polynomial features of price data
3. **Market conditions**: Model trained on historical data may not adapt to changing market conditions
4. **No fundamental analysis**: Only technical price-based features used

## Future Improvements

1. **Additional features**: Include volume, volatility, technical indicators
2. **Model ensemble**: Combine multiple models for better predictions
3. **Online learning**: Implement model updating with new data
4. **Advanced preprocessing**: Add more sophisticated feature engineering
5. **Risk management**: Implement dynamic position sizing and stop losses

## License

This project is based on the MQL5 article by Yevgeniy Koshtenko. Please refer to the original article for licensing terms: https://www.mql5.com/en/articles/13725

## Support

For questions and issues:
1. Check the troubleshooting section above
2. Review the original MQL5 article
3. Examine the code comments and documentation

## Disclaimer

This software is for educational and research purposes only. Trading financial instruments carries risk, and past performance does not guarantee future results. Always test thoroughly in a demo environment before using with real money. 