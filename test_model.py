import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_model():
    """
    Load the saved Random Forest pipeline
    """
    if not os.path.exists('models/random_forest_pipeline.pkl'):
        print("Error: Model file not found. Please run random_forest_model.py first.")
        return None
    
    pipeline = joblib.load('models/random_forest_pipeline.pkl')
    print("Model loaded successfully!")
    return pipeline

def load_model_info():
    """
    Load model information
    """
    if not os.path.exists('models/model_info.json'):
        print("Warning: Model info file not found.")
        return None
    
    with open('models/model_info.json', 'r') as f:
        model_info = json.load(f)
    
    return model_info

def predict_price(pipeline, recent_prices, n_steps=100):
    """
    Predict next price given recent prices
    """
    if len(recent_prices) != n_steps:
        raise ValueError(f"Expected {n_steps} recent prices, got {len(recent_prices)}")
    
    # Reshape for prediction
    X_pred = np.array(recent_prices).reshape(1, -1)
    
    # Make prediction
    prediction = pipeline.predict(X_pred)
    
    return prediction[0]

def generate_sample_data(n_points=100):
    """
    Generate sample EURUSD-like price data for testing
    """
    np.random.seed(42)
    
    # Start with a realistic EURUSD price
    start_price = 1.1000
    
    # Generate realistic price movements
    price_changes = np.random.normal(0, 0.0001, n_points)
    prices = np.cumsum(price_changes) + start_price
    
    return prices

def main():
    print("=== Random Forest Model Tester ===\n")
    
    # Load the model
    pipeline = load_model()
    if pipeline is None:
        return
    
    # Load model information
    model_info = load_model_info()
    if model_info:
        print("Model Information:")
        print(f"- Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"- Time Steps: {model_info.get('n_steps', 'Unknown')}")
        print(f"- Training Size: {model_info.get('training_size', 'Unknown')}")
        print(f"- Test Size: {model_info.get('test_size', 'Unknown')}")
        if model_info.get('r2_score'):
            print(f"- R² Score: {model_info.get('r2_score', 'Unknown'):.4f}")
        if model_info.get('mse'):
            print(f"- MSE: {model_info.get('mse', 'Unknown'):.8f}")
        print()
    
    # Generate sample data for testing
    n_steps = model_info.get('n_steps', 100) if model_info else 100
    print(f"Generating {n_steps} sample price points for testing...")
    
    sample_prices = generate_sample_data(n_steps)
    
    print(f"Sample prices range: {sample_prices.min():.5f} to {sample_prices.max():.5f}")
    
    # Make a prediction
    try:
        predicted_price = predict_price(pipeline, sample_prices, n_steps)
        print(f"Predicted next price: {predicted_price:.5f}")
        print(f"Last actual price: {sample_prices[-1]:.5f}")
        print(f"Predicted change: {predicted_price - sample_prices[-1]:.5f}")
        
        # Determine direction
        if predicted_price > sample_prices[-1]:
            direction = "UP ↗"
        elif predicted_price < sample_prices[-1]:
            direction = "DOWN ↘"
        else:
            direction = "SIDEWAYS →"
        
        print(f"Predicted direction: {direction}")
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return
    
    # Create visualization
    print("\nCreating visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot the sample prices
    plt.subplot(2, 1, 1)
    plt.plot(sample_prices, label='Sample Prices', color='blue', alpha=0.7)
    plt.axhline(y=predicted_price, color='red', linestyle='--', 
                label=f'Predicted Next Price: {predicted_price:.5f}')
    plt.title('Sample Price Data and Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot price changes
    plt.subplot(2, 1, 2)
    price_changes = np.diff(sample_prices)
    plt.plot(price_changes, label='Price Changes', color='green', alpha=0.7)
    plt.title('Price Changes')
    plt.xlabel('Time Steps')
    plt.ylabel('Price Change')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_test_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'model_test_visualization.png'")
    plt.close()
    
    # Test multiple predictions
    print("\n🔄 Testing multiple predictions...")
    predictions = []
    actuals = []
    
    # Generate a longer sequence for testing
    test_sequence = generate_sample_data(n_steps + 50)
    
    # Add progress bar for multiple predictions
    with tqdm(total=len(test_sequence) - n_steps, desc="Testing predictions", unit="predictions") as pbar:
        for i in range(n_steps, len(test_sequence)):
            recent_prices = test_sequence[i-n_steps:i]
            predicted = predict_price(pipeline, recent_prices, n_steps)
            actual = test_sequence[i]
            
            predictions.append(predicted)
            actuals.append(actual)
            pbar.update(1)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate test metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"Test Results on {len(predictions)} predictions:")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"RMSE: {np.sqrt(mse):.8f}")
    
    # Direction accuracy
    pred_direction = np.sign(np.diff(predictions))
    actual_direction = np.sign(np.diff(actuals))
    direction_accuracy = np.mean(pred_direction == actual_direction) * 100
    
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    
    print("\n=== Model Test Completed Successfully ===")

if __name__ == "__main__":
    main() 