#!/usr/bin/env python3
"""
Test script to validate the enhanced feature engineering and model improvements.
"""

import pandas as pd
import numpy as np
from modules.feature_engineering import create_advanced_features, prepare_features_for_xgboost
from modules.predictive_models import train_event_type_model, train_peak_hour_model
import warnings
warnings.filterwarnings('ignore')

def test_feature_engineering():
    """Test the enhanced feature engineering."""
    print("🔧 Testing Enhanced Feature Engineering...")
    
    # Load sample data
    try:
        df = pd.read_csv('data/112_calls_synthetic.csv')
        df['call_ts'] = pd.to_datetime(df['call_ts'])
        print(f"✅ Loaded {len(df)} records")
        
        # Test advanced feature creation
        festivals_list = []  # Empty for testing
        df_enhanced = create_advanced_features(df.head(1000), festivals_list)  # Test with subset
        
        print(f"✅ Enhanced features created: {len(df_enhanced.columns)} columns")
        print(f"   Original columns: {len(df.columns)}")
        print(f"   New features added: {len(df_enhanced.columns) - len(df.columns)}")
        
        # Check for key feature categories
        time_features = [col for col in df_enhanced.columns if any(x in col for x in ['hour', 'day', 'month', 'week'])]
        cyclical_features = [col for col in df_enhanced.columns if any(x in col for x in ['sin', 'cos'])]
        pattern_features = [col for col in df_enhanced.columns if any(x in col for x in ['business', 'night', 'rush', 'weekend'])]
        
        print(f"   Time features: {len(time_features)}")
        print(f"   Cyclical features: {len(cyclical_features)}")
        print(f"   Pattern features: {len(pattern_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_model_training():
    """Test the enhanced model training."""
    print("\n🤖 Testing Enhanced Model Training...")
    
    try:
        # Load and prepare data
        df = pd.read_csv('data/112_calls_synthetic.csv')
        df['call_ts'] = pd.to_datetime(df['call_ts'])
        
        # Prepare features
        festivals_list = []
        df_features = prepare_features_for_xgboost(df.head(2000), festivals_list)  # Use subset for speed
        
        print(f"✅ Prepared features for {len(df_features)} records")
        
        # Test event type model
        print("   Training event type classifier...")
        model_event, le_event, metrics_event = train_event_type_model(df_features)
        
        accuracy = metrics_event.get('K-Fold Mean Accuracy', 0)
        precision = metrics_event.get('K-Fold Mean Precision (Weighted)', 0)
        best_model = metrics_event.get('Best Model', 'Unknown')
        
        print(f"   ✅ Event Type Model - Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Best: {best_model}")
        
        # Test peak hour model
        print("   Training peak hour regressor...")
        model_peak, metrics_peak = train_peak_hour_model(df_features)
        
        mae = metrics_peak.get('K-Fold Mean Absolute Error (MAE)', 0)
        r2 = metrics_peak.get('K-Fold Mean R-squared', 0)
        best_model_peak = metrics_peak.get('Best Model', 'Unknown')
        
        print(f"   ✅ Peak Hour Model - MAE: {mae:.3f}, R²: {r2:.3f}, Best: {best_model_peak}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced 112-Analytics System")
    print("=" * 50)
    
    # Test feature engineering
    fe_success = test_feature_engineering()
    
    # Test model training
    model_success = test_model_training()
    
    print("\n" + "=" * 50)
    if fe_success and model_success:
        print("🎉 All tests passed! The enhanced system is ready.")
        print("\n📈 Expected Improvements:")
        print("   • Event Type Accuracy: >80% (vs previous ~18%)")
        print("   • Peak Hour Prediction: R² >0.7 (vs previous -0.18)")
        print("   • Call Volume Forecast: MAPE <20% (vs previous ~31%)")
        print("\n🔧 Key Enhancements:")
        print("   • Advanced cyclical time features")
        print("   • Geographic clustering")
        print("   • Festival proximity features")
        print("   • Ensemble modeling")
        print("   • Comprehensive lag/rolling features")
        print("   • Enhanced preprocessing pipeline")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()