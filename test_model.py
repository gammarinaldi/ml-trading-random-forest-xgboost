import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_unified_model():
    """
    Load the unified price regressor model
    """
    if os.path.exists('models/unified_price_regressor.pkl'):
        model = joblib.load('models/unified_price_regressor.pkl')
        print("Unified price regressor loaded successfully!")
        return model
    else:
        print("Unified price regressor not found. Run the updated training script first.")
        return None

def load_unified_model_info():
    """
    Load unified model information
    """
    if not os.path.exists('models/unified_price_model_info.json'):
        print("Warning: Unified model info file not found.")
        return None
    
    with open('models/unified_price_model_info.json', 'r') as f:
        model_info = json.load(f)
    
    return model_info

def add_technical_indicators(df, price_col='close'):
    """
    Add the same technical indicators used in training
    """
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

def predict_price_and_direction(model, recent_data, feature_cols, n_steps=100, threshold=0.0001):
    """
    Predict price and derive direction using the unified model
    """
    if len(recent_data) < n_steps:
        raise ValueError(f"Need at least {n_steps} data points, got {len(recent_data)}")
    
    # Get the last n_steps of features
    features = recent_data[feature_cols].iloc[-n_steps:].values.flatten()
    features = features.reshape(1, -1)
    
    # Make price prediction
    price_pred = model.predict(features)[0]
    
    # Derive direction from price prediction
    current_price = recent_data['close'].iloc[-1]
    price_change = (price_pred - current_price) / current_price
    
    # Determine direction based on price change (no arbitrary defaults)
    direction_pred = 1 if price_change > 0 else 0
    
    # Confidence based on magnitude of change
    if abs(price_change) > threshold:
        confidence = abs(price_change)
    else:
        confidence = abs(price_change) * 0.1  # Lower confidence for small changes
    
    return direction_pred, price_pred, confidence

def generate_realistic_sample_data(n_points=200, start_price=1.10000):
    """
    Generate more realistic EURUSD-like price data with trends
    """
    np.random.seed(42)
    
    # Create price series with trend and noise
    trend = np.linspace(0, 0.005, n_points)  # Slight upward trend
    noise = np.random.normal(0, 0.0002, n_points)
    cyclical = 0.001 * np.sin(np.linspace(0, 4*np.pi, n_points))
    
    prices = start_price + trend + cyclical + np.cumsum(noise)
    
    # Create DataFrame with timestamp
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='h')
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    return df

def main():
    print("=== Unified Price Model Tester ===\n")
    
    # Load the unified model
    model = load_unified_model()
    if model is None:
        print("Cannot proceed without model. Please run the updated training script first.")
        return
    
    # Load model information
    model_info = load_unified_model_info()
    if model_info:
        print("Unified Model Information:")
        print(f"Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"Time Steps: {model_info.get('n_steps', 'Unknown')}")
        print(f"Features: {model_info.get('n_features', 'Unknown')}")
        print(f"Training Size: {model_info.get('training_size', 'Unknown'):,}")
        print(f"Test Size: {model_info.get('test_size', 'Unknown'):,}")
        print(f"Price R²: {model_info.get('price_r2', 0):.4f}")
        print(f"Direction Accuracy (derived): {model_info.get('direction_accuracy_derived', 0)*100:.2f}%")
        print(f"Approach: {model_info.get('approach', 'Unknown')}")
        print()
    
    # Generate realistic sample data
    n_steps = model_info.get('n_steps', 100) if model_info else 100
    print(f"Generating realistic sample data for testing...")
    
    sample_data = generate_realistic_sample_data(n_steps + 100)
    
    # Add technical indicators
    print("Adding technical indicators...")
    sample_data = add_technical_indicators(sample_data, 'close')
    sample_data = sample_data.dropna()  # Remove NaN values from indicators
    
    print(f"Sample data prepared: {len(sample_data)} records")
    print(f"Price range: {sample_data['close'].min():.5f} to {sample_data['close'].max():.5f}")
    
    # Get feature columns (exclude close)
    feature_cols = [col for col in sample_data.columns if col not in ['close']]
    
    if len(sample_data) < n_steps:
        print(f"Not enough data for prediction. Need {n_steps}, got {len(sample_data)}")
        return
    
    # Make a single prediction
    print(f"\nMaking unified prediction...")
    try:
        direction_pred, price_pred, confidence = predict_price_and_direction(
            model, sample_data, feature_cols, n_steps
        )
        
        current_price = sample_data['close'].iloc[-1]
        direction_names = {0: "DOWN", 1: "UP"}
        
        print(f"Unified Prediction Results:")
        print(f"Current Price: {current_price:.5f}")
        print(f"Predicted Price: {price_pred:.5f}")
        print(f"Derived Direction: {direction_names.get(direction_pred, 'Unknown')}")
        print(f"Predicted Change: {price_pred - current_price:.5f}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Show consistency check
        actual_direction = 1 if price_pred > current_price else 0
        print(f"Direction-Price Consistency: {'CONSISTENT' if direction_pred == actual_direction else 'INCONSISTENT'}")
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return
    
    # Test multiple predictions
    print(f"\nTesting rolling predictions...")
    
    n_test_predictions = min(50, len(sample_data) - n_steps - 1)
    predictions = []
    actuals = []
    direction_preds = []
    direction_actuals = []
    
    # Store original indices to fix consistency calculation
    prediction_indices = []
    
    with tqdm(total=n_test_predictions, desc="Rolling predictions", unit="predictions") as pbar:
        for i in range(n_test_predictions):
            start_idx = i
            end_idx = start_idx + n_steps
            
            # Get data slice
            data_slice = sample_data.iloc[start_idx:end_idx]
            
            # Make prediction
            try:
                dir_pred, price_pred, conf = predict_price_and_direction(
                    model, data_slice, feature_cols, n_steps
                )
                
                # Get actual values (next time step)
                actual_price = sample_data['close'].iloc[end_idx]
                current_price = sample_data['close'].iloc[end_idx-1]
                
                # Calculate actual direction
                price_change = (actual_price - current_price) / current_price
                # Only include clear directional movements (matching training logic)
                if abs(price_change) > 0.0001:
                    actual_direction = 1 if price_change > 0.0001 else 0
                    
                    predictions.append(price_pred)
                    actuals.append(actual_price)
                    direction_preds.append(dir_pred)
                    direction_actuals.append(actual_direction)
                    prediction_indices.append(i)  # Store original index
                
            except Exception as e:
                print(f"Warning: Prediction {i} failed: {str(e)}")
                
            pbar.update(1)
    
    # Calculate unified metrics
    if len(predictions) > 0:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        direction_preds = np.array(direction_preds)
        direction_actuals = np.array(direction_actuals)
        
        # Price prediction metrics
        price_mse = np.mean((predictions - actuals) ** 2)
        price_mae = np.mean(np.abs(predictions - actuals))
        price_rmse = np.sqrt(price_mse)
        
        # Direction accuracy
        direction_accuracy = np.mean(direction_preds == direction_actuals) * 100
        
        print(f"\nUnified Test Results on {len(predictions)} predictions:")
        print(f"Price Prediction:")
        print(f"MSE: {price_mse:.8f}")
        print(f"MAE: {price_mae:.6f}")
        print(f"RMSE: {price_rmse:.6f}")
        print(f"Average pip error: {price_mae/np.mean(actuals)*10000:.1f} pips")
        
        print(f"Direction Prediction (derived from price):")
        print(f"Accuracy: {direction_accuracy:.2f}%")

        # Base rate: always predicting the majority class. On synthetic data with an
        # embedded trend this is a high bar, and accuracy must be read against it.
        base_rate = max(np.mean(direction_actuals == 1), np.mean(direction_actuals == 0)) * 100
        print(f"Majority-class base rate: {base_rate:.2f}%")
        print(f"Edge over base rate: {direction_accuracy - base_rate:+.2f} points")
        
        # Direction breakdown
        for direction in [0, 1]:
            mask = direction_actuals == direction
            if np.sum(mask) > 0:
                acc = np.mean(direction_preds[mask] == direction_actuals[mask]) * 100
                direction_name = {0: "DOWN", 1: "UP"}[direction]
                print(f"{direction_name} accuracy: {acc:.1f}% ({np.sum(mask)} samples)")
        
        # Consistency check - should always be 100% by design
        # Use same logic as predict_price_and_direction function with correct indices
        threshold = 0.0001
        consistent_directions = []
        for idx, orig_i in enumerate(prediction_indices):
            # Use the correct current_price from when the prediction was made
            current_price = sample_data['close'].iloc[orig_i + n_steps - 1]
            price_change = (predictions[idx] - current_price) / current_price
            
            # Use same logic as predict_price_and_direction (no arbitrary defaults)
            expected_dir = 1 if price_change > 0 else 0
            
            consistent_directions.append(expected_dir)
        
        consistency = np.mean(direction_preds == np.array(consistent_directions)) * 100
        print(f"Direction-Price Consistency: {consistency:.1f}% (should be 100%)")
        
        # Debug: Show small vs large price changes
        small_changes = sum(1 for idx, orig_i in enumerate(prediction_indices) 
                           if abs((predictions[idx] - sample_data['close'].iloc[orig_i + n_steps - 1]) / 
                                 sample_data['close'].iloc[orig_i + n_steps - 1]) <= threshold)
        print(f"Small price changes (<={threshold:.4f}): {small_changes}/{len(predictions)} ({small_changes/len(predictions)*100:.1f}%)")
    
    # Create unified visualization
    print(f"\nCreating unified visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price predictions vs actual
    if len(predictions) > 0:
        axes[0,0].plot(actuals, label='Actual Prices', alpha=0.7, color='blue')
        axes[0,0].plot(predictions, label='Predicted Prices', alpha=0.7, color='red')
        axes[0,0].set_title('Unified Price Predictions vs Actual')
        axes[0,0].set_xlabel('Time Steps')
        axes[0,0].set_ylabel('Price')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Prediction errors
        errors = predictions - actuals
        axes[0,1].plot(errors, alpha=0.7, color='orange')
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0,1].set_title('Price Prediction Errors')
        axes[0,1].set_xlabel('Time Steps')
        axes[0,1].set_ylabel('Error')
        axes[0,1].grid(True)
        
        # Direction accuracy over time (derived from price)
        direction_correct = (direction_preds == direction_actuals).astype(int)
        rolling_dir_acc = pd.Series(direction_correct).rolling(window=10, min_periods=1).mean()
        axes[1,0].plot(rolling_dir_acc * 100, color='green', alpha=0.8)
        axes[1,0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        axes[1,0].set_title('Rolling Direction Accuracy (Derived from Price)')
        axes[1,0].set_xlabel('Time Steps')
        axes[1,0].set_ylabel('Accuracy (%)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Sample price data with indicators.
        # Plot against positional integers for every series: the index is a DatetimeIndex,
        # and mixing datetime x-values with integer positions silently breaks the axis.
        sample_slice = sample_data.tail(100)
        x_sample = np.arange(len(sample_slice))
        axes[1,1].plot(x_sample, sample_slice['close'].to_numpy(), label='Close Price', color='blue')
        axes[1,1].plot(x_sample, sample_slice['sma_20'].to_numpy(), label='SMA 20', color='orange', alpha=0.7)
        axes[1,1].plot(x_sample, sample_slice['ema_10'].to_numpy(), label='EMA 10', color='green', alpha=0.7)
        axes[1,1].fill_between(x_sample,
                              sample_slice['bb_lower'].to_numpy(), sample_slice['bb_upper'].to_numpy(),
                              alpha=0.2, color='gray', label='Bollinger Bands')
        axes[1,1].set_title('Sample Data with Technical Indicators')
        axes[1,1].set_xlabel('Time Steps')
        axes[1,1].set_ylabel('Price')
        axes[1,1].legend()
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('unified_model_test_results.png', dpi=300, bbox_inches='tight')
    print("Unified visualization saved as 'unified_model_test_results.png'")
    plt.close()
    
    print(f"\nUnified model testing completed!")
    
    if len(predictions) > 0:
        print(f"NOTE: this script evaluates on SYNTHETIC data generated by")
        print(f"generate_realistic_sample_data(), which embeds an upward trend. Accuracy here")
        print(f"measures nothing about real EURUSD and must not be quoted as a result. It is a")
        print(f"smoke test that the model loads and produces consistent output. For the real")
        print(f"hold-out numbers, see models/unified_price_model_info.json and the README.")
        
        if direction_accuracy > base_rate:
            print(f"Direction accuracy ({direction_accuracy:.1f}%) exceeds the base rate "
                  f"({base_rate:.1f}%) on this synthetic sample, which is expected given the")
            print(f"embedded trend and the 30 usable samples - not evidence of an edge.")
        else:
            print(f"Direction accuracy ({direction_accuracy:.1f}%) does not exceed the base rate "
                  f"({base_rate:.1f}%) even on trended synthetic data.")
        
        print(f"Direction and price predictions are internally consistent by construction,")
        print(f"since direction is derived from the sign of the predicted price change.")

if __name__ == "__main__":
    main() 