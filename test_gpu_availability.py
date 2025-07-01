#!/usr/bin/env python3
"""
GPU Availability Test for Machine Learning
Tests various GPU acceleration options for your forex model
"""

import sys
import numpy as np

def test_xgboost_gpu():
    """Test XGBoost GPU availability"""
    try:
        import xgboost as xgb
        print("✅ XGBoost installed")
        
        # Test GPU functionality
        try:
            test_data = xgb.DMatrix(np.random.random((100, 10)))
            gpu_params = {
                'tree_method': 'hist',
                'device': 'cuda',
                'objective': 'reg:squarederror'
            }
            xgb.train(gpu_params, test_data, num_boost_round=5, verbose_eval=False)
            print("🚀 XGBoost GPU: ✅ AVAILABLE")
            return True
        except Exception as e:
            print(f"❌ XGBoost GPU: Not available ({str(e)[:50]}...)")
            print("💻 XGBoost CPU: ✅ Available")
            return False
            
    except ImportError:
        print("❌ XGBoost: Not installed")
        print("   Install with: pip install xgboost")
        return False

def test_rapids_cuml():
    """Test RAPIDS cuML availability"""
    try:
        import cuml
        import cudf
        print("✅ RAPIDS cuML installed")
        print("🚀 RAPIDS GPU: ✅ AVAILABLE")
        return True
    except ImportError:
        print("❌ RAPIDS cuML: Not installed")
        print("   Install with: conda install -c rapidsai -c conda-forge cuml")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU availability"""
    try:
        import tensorflow as tf
        print("✅ TensorFlow installed")
        
        if tf.config.list_physical_devices('GPU'):
            print("🚀 TensorFlow GPU: ✅ AVAILABLE")
            print(f"   GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
            return True
        else:
            print("❌ TensorFlow GPU: Not available")
            print("💻 TensorFlow CPU: ✅ Available")
            return False
            
    except ImportError:
        print("❌ TensorFlow: Not installed")
        print("   Install with: pip install tensorflow")
        return False

def test_lightgbm_gpu():
    """Test LightGBM GPU availability"""
    try:
        import lightgbm as lgb
        print("✅ LightGBM installed")
        
        # Create test data
        train_data = lgb.Dataset(np.random.random((100, 10)), 
                                label=np.random.random(100))
        
        try:
            # Try GPU training
            gpu_params = {
                'objective': 'regression',
                'device': 'gpu',
                'verbose': -1
            }
            lgb.train(gpu_params, train_data, num_boost_round=5, verbose_eval=False)
            print("🚀 LightGBM GPU: ✅ AVAILABLE")
            return True
        except Exception:
            print("❌ LightGBM GPU: Not available")
            print("💻 LightGBM CPU: ✅ Available")
            return False
            
    except ImportError:
        print("❌ LightGBM: Not installed")
        print("   Install with: pip install lightgbm")
        return False

def main():
    print("🎮 GPU Availability Test for Forex ML Models")
    print("=" * 50)
    
    gpu_available = False
    
    print("\n1️⃣ Testing XGBoost (Recommended - easiest GPU setup):")
    if test_xgboost_gpu():
        gpu_available = True
    
    print("\n2️⃣ Testing RAPIDS cuML (Pure Random Forest GPU):")
    if test_rapids_cuml():
        gpu_available = True
    
    print("\n3️⃣ Testing TensorFlow (Deep Learning + Decision Forests):")
    if test_tensorflow_gpu():
        gpu_available = True
    
    print("\n4️⃣ Testing LightGBM (Alternative Gradient Boosting):")
    if test_lightgbm_gpu():
        gpu_available = True
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    
    if gpu_available:
        print("🎉 GPU acceleration is available!")
        print("\n🚀 To use GPU with your model:")
        print("   python improved_directional_model.py")
        print("   # Will automatically use GPU if available")
        
        print("\n💡 For maximum performance:")
        print("   • XGBoost GPU: Best balance of speed and ease")
        print("   • RAPIDS cuML: True Random Forest on GPU")
        print("   • Your model will automatically detect and use GPU")
    else:
        print("❌ No GPU acceleration detected")
        print("\n💻 Your model will use CPU (still fast with multiple cores)")
        print("\n🔧 To enable GPU:")
        print("   1. Install NVIDIA GPU with CUDA support")
        print("   2. Install: pip install xgboost")
        print("   3. Run: python test_gpu_availability.py")
    
    print("\n📁 Available model scripts:")
    print("   • improved_directional_model.py (auto GPU/CPU)")
    print("   • improved_directional_model_gpu.py (RAPIDS cuML)")
    print("   • xgboost_gpu_model.py (XGBoost optimized)")

if __name__ == "__main__":
    main() 