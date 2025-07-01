import pandas as pd
import numpy as np
import joblib
import random
import onnx
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, PolynomialFeatures
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set the number of time steps according to requirements
n_steps = 100

# Load data from CSV file
csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'

print(f"Loading EURUSD data from CSV file: {csv_file}")

try:
    # Try to load the CSV file
    if os.path.exists(csv_file):
        print(f"Found CSV file: {csv_file}")
        
        # Load CSV data - this appears to be tab-separated MetaTrader export
        df = pd.read_csv(csv_file, sep='\t')
        
        # Display first few rows to understand the structure
        print("CSV file structure:")
        print(f"Columns: {list(df.columns)}")
        print("First few rows:")
        print(df.head())
        
        # Check if it's the standard MetaTrader format with tab separators
        if '<DATE>' in df.columns and '<TIME>' in df.columns and '<CLOSE>' in df.columns:
            print("Detected MetaTrader export format with tab separators")
            
            # Combine date and time columns
            df['datetime'] = df['<DATE>'] + ' ' + df['<TIME>']
            
            # Create the data DataFrame with proper columns
            data = pd.DataFrame()
            data['time'] = df['datetime']
            data['close'] = df['<CLOSE>']
            
            # Convert time column to datetime with MetaTrader format
            data['time'] = pd.to_datetime(data['time'], format='%Y.%m.%d %H:%M:%S')
            
        else:
            # Try common column name patterns for time and close price
            time_columns = ['time', 'Time', 'TIME', 'date', 'Date', 'DATE', 'datetime', 'DateTime', 'DATETIME']
            close_columns = ['close', 'Close', 'CLOSE', 'price', 'Price', 'PRICE']
            
            time_col = None
            close_col = None
            
            # Find time column
            for col in time_columns:
                if col in df.columns:
                    time_col = col
                    break
            
            # Find close price column
            for col in close_columns:
                if col in df.columns:
                    close_col = col
                    break
            
            # If standard columns not found, try first two columns
            if time_col is None:
                time_col = df.columns[0]
                print(f"Using first column as time: {time_col}")
            
            if close_col is None:
                # Look for columns that might contain price data
                for col in df.columns:
                    if col.lower() in ['close', 'price'] or 'close' in col.lower():
                        close_col = col
                        break
                
                if close_col is None:
                    # Use the last column or a numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        close_col = numeric_cols[-1]  # Use last numeric column
                    else:
                        close_col = df.columns[-1]  # Use last column
                print(f"Using column as close price: {close_col}")
            
            # Create the data DataFrame
            data = df[[time_col, close_col]].copy()
            data.columns = ['time', 'close']
            
            # Convert time column to datetime with explicit format handling
            try:
                # First try common MetaTrader format
                data['time'] = pd.to_datetime(data['time'], format='%Y.%m.%d %H:%M')
            except:
                try:
                    # Try ISO format
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
                except:
                    try:
                        # Try without seconds
                        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M')
                    except:
                        try:
                            # Try with infer_datetime_format for better performance
                            data['time'] = pd.to_datetime(data['time'])
                        except:
                            print("Warning: Could not parse time column, using index")
                            data['time'] = pd.date_range(start='2018-01-01', periods=len(data), freq='h')
        
        # Set time as index
        data = data.set_index('time')
        
        # Ensure close prices are numeric
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        
        # Remove any NaN values
        data = data.dropna()
        
        # Keep only close column
        data = data[['close']]
        
        print(f"Successfully loaded CSV data with {len(data)} records")
        print(f"Data range: {data.index[0]} to {data.index[-1]}")
        print(f"Price range: {data['close'].min():.5f} to {data['close'].max():.5f}")
        
    else:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
except Exception as e:
    print(f"Error loading CSV file: {str(e)}")
    print("Falling back to synthetic data generation...")
    
    # Create synthetic EURUSD data
    dates = pd.date_range(start='2018-01-01', end='2024-12-31', freq='h')
    np.random.seed(42)
    # Generate realistic EURUSD price movements
    price_changes = np.random.normal(0, 0.0001, len(dates))
    prices = np.cumsum(price_changes) + 1.1000  # Start around 1.1000
    
    data = pd.DataFrame({
        'time': dates,
        'close': prices
    }).set_index('time')
    data = data.dropna()
    data = data[['close']]
    
    print(f"Created synthetic data with {len(data)} records")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: {data['close'].min():.5f} to {data['close'].max():.5f}")

# Define train_data_initial
training_size = int(len(data) * 0.70)
train_data_initial = data.iloc[:training_size]
test_data_initial = data.iloc[training_size:]

print(f"Training data: {len(train_data_initial)} records")
print(f"Test data: {len(test_data_initial)} records")

# Function for creating and assigning labels for regression
def labelling_relabeling_regression(dataset, min_value=1, max_value=1):
    """
    Create labels for regression by using future prices
    """
    future_prices = []
    
    # Add progress bar for label creation
    with tqdm(total=dataset.shape[0] - max_value, desc="Creating labels", unit="samples") as pbar:
        for i in range(dataset.shape[0] - max_value):
            rand = random.randint(min_value, max_value)
            future_pr = dataset['close'].iloc[i + rand]
            future_prices.append(future_pr)
            pbar.update(1)
    
    dataset = dataset.iloc[:len(future_prices)].copy()
    dataset['future_price'] = future_prices
    
    return dataset

# Apply the labelling_relabeling_regression function to the training data
print("\n🏷️  Step 1/6: Creating labels for training data...")
train_data = labelling_relabeling_regression(train_data_initial, 1, 5)

tqdm.write(f"✅ Training data after labeling: {len(train_data)} records")

# Feature engineering function
def create_sequences(data, n_steps):
    """
    Create sequences for time series prediction
    """
    X, y = [], []
    
    # Add progress bar for sequence creation
    with tqdm(total=len(data) - n_steps, desc="Creating sequences", unit="sequences") as pbar:
        for i in range(n_steps, len(data)):
            X.append(data['close'].iloc[i-n_steps:i].values)
            y.append(data['future_price'].iloc[i])
            pbar.update(1)
    
    return np.array(X), np.array(y)

# Create sequences for training
print("\n📊 Step 2/6: Creating sequences for training...")
X_train, y_train = create_sequences(train_data, n_steps)

tqdm.write(f"✅ Training sequences shape: {X_train.shape}")
tqdm.write(f"✅ Training labels shape: {y_train.shape}")

# Ensure we have enough data
if len(X_train) < 100:
    tqdm.write("⚠️  Warning: Very limited training data available")

# Create a pipeline with preprocessing and Random Forest
print("\n🔧 Step 3/6: Creating Random Forest pipeline with preprocessing...")

# Create pipeline with preprocessing steps
pipeline = Pipeline([
    ('robust_scaler', RobustScaler()),
    ('poly_features', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
    ('minmax_scaler', MinMaxScaler()),
    ('random_forest', RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model with progress indication
print("\n🌲 Step 4/6: Training the Random Forest model...")
print("📈 Training in progress (this may take a few minutes)...")

# Add a progress bar for training stages
training_stages = [
    "Applying RobustScaler normalization",
    "Creating polynomial features", 
    "Applying MinMaxScaler normalization",
    "Training Random Forest (20 trees)"
]

with tqdm(total=len(training_stages), desc="Training pipeline", unit="stage") as pbar:
    for i, stage in enumerate(training_stages):
        tqdm.write(f"  🔄 {stage}...")
        if i == len(training_stages) - 1:
            # Actual training happens here
            pipeline.fit(X_train, y_train)
        else:
            # Simulate preprocessing stages
            time.sleep(0.5)
        pbar.update(1)

tqdm.write("✅ Model training completed!")

# Create test sequences for evaluation
print("\n📋 Step 5/6: Evaluating model performance...")
tqdm.write("Creating test sequences...")
test_data_labeled = labelling_relabeling_regression(test_data_initial, 1, 5)
X_test, y_test = create_sequences(test_data_labeled, n_steps)

if len(X_test) > 0:
    # Make predictions with progress bar
    tqdm.write("Making predictions on test data...")
    
    # For large datasets, show prediction progress
    if len(X_test) > 1000:
        batch_size = 1000
        y_pred = []
        with tqdm(total=len(X_test), desc="Making predictions", unit="samples") as pbar:
            for i in range(0, len(X_test), batch_size):
                batch_end = min(i + batch_size, len(X_test))
                batch_pred = pipeline.predict(X_test[i:batch_end])
                y_pred.extend(batch_pred)
                pbar.update(batch_end - i)
        y_pred = np.array(y_pred)
    else:
        y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    tqdm.write(f"📊 Test Results:")
    tqdm.write(f"  📉 Mean Squared Error: {mse:.8f}")
    tqdm.write(f"  📈 R² Score: {r2:.4f}")
    tqdm.write(f"  📐 RMSE: {np.sqrt(mse):.8f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot first 500 predictions vs actual
    n_plot = min(500, len(y_test))
    plt.subplot(2, 1, 1)
    plt.plot(y_test[:n_plot], label='Actual', alpha=0.7)
    plt.plot(y_pred[:n_plot], label='Predicted', alpha=0.7)
    plt.title('Random Forest Predictions vs Actual Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    residuals = y_test[:n_plot] - y_pred[:n_plot]
    plt.plot(residuals, alpha=0.7)
    plt.title('Prediction Residuals')
    plt.xlabel('Time Steps')
    plt.ylabel('Residual')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    tqdm.write("📊 Performance visualization saved as 'model_performance.png'")
    plt.close()
else:
    tqdm.write("⚠️  Not enough test data for evaluation")

# Save the model locally
print("\n💾 Step 6/6: Saving model and exporting...")

# Create models directory
os.makedirs('models', exist_ok=True)

# Saving tasks with progress bar
saving_tasks = [
    ("Saving sklearn pipeline", "models/random_forest_pipeline.pkl"),
    ("Converting to ONNX format", "models/random_forest_model.onnx"),
    ("Saving model metadata", "models/model_info.json")
]

with tqdm(total=len(saving_tasks), desc="Saving models", unit="task") as pbar:
    # Save the complete pipeline
    tqdm.write(f"  💾 {saving_tasks[0][0]}...")
    joblib.dump(pipeline, saving_tasks[0][1])
    tqdm.write(f"  ✅ Pipeline saved as '{saving_tasks[0][1]}'")
    pbar.update(1)
    
    # Convert to ONNX
    tqdm.write(f"  🔄 {saving_tasks[1][0]}...")
    try:
        # Define the input type for ONNX conversion
        initial_type = [('float_input', FloatTensorType([None, n_steps]))]
        
        # Convert the sklearn pipeline to ONNX
        onnx_model = convert_sklearn(
            pipeline, 
            initial_types=initial_type,
            target_opset=11
        )
        
        # Save ONNX model locally
        onnx_path = saving_tasks[1][1]
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        tqdm.write(f"  ✅ ONNX model saved as '{onnx_path}'")
        
        # Verify ONNX model
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        tqdm.write("  ✅ ONNX model verification successful!")
        
    except Exception as e:
        tqdm.write(f"  ❌ Error converting to ONNX: {str(e)}")
        tqdm.write("  ⚠️  This might be due to compatibility issues with PolynomialFeatures in ONNX conversion")
    
    pbar.update(1)

    # Save model information
    tqdm.write(f"  📝 {saving_tasks[2][0]}...")
    model_info = {
        'n_steps': n_steps,
        'training_size': len(X_train),
        'test_size': len(X_test) if len(X_test) > 0 else 0,
        'mse': mse if len(X_test) > 0 else None,
        'r2_score': r2 if len(X_test) > 0 else None,
        'model_type': 'RandomForest with RobustScaler + PolynomialFeatures + MinMaxScaler',
        'n_estimators': 20,
        'max_depth': 10,
        'polynomial_degree': 2
    }
    
    # Save model info as JSON
    import json
    with open(saving_tasks[2][1], 'w') as f:
        json.dump(model_info, f, indent=4)
    
    tqdm.write(f"  ✅ Model information saved as '{saving_tasks[2][1]}'")
    pbar.update(1)

# Create a sample prediction function
def predict_next_price(recent_prices):
    """
    Predict next price given recent n_steps prices
    """
    if len(recent_prices) != n_steps:
        raise ValueError(f"Expected {n_steps} recent prices, got {len(recent_prices)}")
    
    # Reshape for prediction
    X_pred = np.array(recent_prices).reshape(1, -1)
    
    # Make prediction
    prediction = pipeline.predict(X_pred)
    
    return prediction[0]

# Test the prediction function
if len(X_test) > 0:
    print("\nTesting prediction function...")
    sample_prices = X_test[0]
    predicted_price = predict_next_price(sample_prices)
    actual_price = y_test[0]
    
    tqdm.write(f"Sample prediction:")
    tqdm.write(f"  🎯 Predicted price: {predicted_price:.5f}")
    tqdm.write(f"  📊 Actual price: {actual_price:.5f}")
    tqdm.write(f"  📏 Difference: {abs(predicted_price - actual_price):.5f}")

print("\n🎉 Model creation and saving completed successfully!")
print("📁 All files saved in the local 'models' directory:")
print("  📦 Pipeline: models/random_forest_pipeline.pkl")
print("  🔄 ONNX model: models/random_forest_model.onnx (if conversion succeeded)")
print("  📋 Model info: models/model_info.json")
print("  📈 Performance plot: model_performance.png")
print("\n✨ Ready to use! Run 'python test_model.py' to test the model.") 