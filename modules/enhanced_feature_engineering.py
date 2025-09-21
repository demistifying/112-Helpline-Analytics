# Enhanced Feature Engineering for Maximum Performance
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

def create_advanced_temporal_features(df):
    """Create advanced temporal features for maximum performance"""
    df = df.copy()
    df['call_ts'] = pd.to_datetime(df['call_ts'])
    
    # Enhanced cyclical features with multiple harmonics
    df['hour_sin1'] = np.sin(2 * np.pi * df['call_ts'].dt.hour / 24)
    df['hour_cos1'] = np.cos(2 * np.pi * df['call_ts'].dt.hour / 24)
    df['hour_sin2'] = np.sin(4 * np.pi * df['call_ts'].dt.hour / 24)
    df['hour_cos2'] = np.cos(4 * np.pi * df['call_ts'].dt.hour / 24)
    
    # Day of week with multiple harmonics
    df['dow_sin1'] = np.sin(2 * np.pi * df['call_ts'].dt.dayofweek / 7)
    df['dow_cos1'] = np.cos(2 * np.pi * df['call_ts'].dt.dayofweek / 7)
    df['dow_sin2'] = np.sin(4 * np.pi * df['call_ts'].dt.dayofweek / 7)
    df['dow_cos2'] = np.cos(4 * np.pi * df['call_ts'].dt.dayofweek / 7)
    
    # Month with multiple harmonics
    df['month_sin1'] = np.sin(2 * np.pi * df['call_ts'].dt.month / 12)
    df['month_cos1'] = np.cos(2 * np.pi * df['call_ts'].dt.month / 12)
    df['month_sin2'] = np.sin(4 * np.pi * df['call_ts'].dt.month / 12)
    df['month_cos2'] = np.cos(4 * np.pi * df['call_ts'].dt.month / 12)
    
    # Advanced time-based features
    df['is_business_hour'] = ((df['call_ts'].dt.hour >= 9) & (df['call_ts'].dt.hour <= 17)).astype(int)
    df['is_peak_hour'] = ((df['call_ts'].dt.hour >= 18) & (df['call_ts'].dt.hour <= 21)).astype(int)
    df['is_night'] = ((df['call_ts'].dt.hour >= 22) | (df['call_ts'].dt.hour <= 5)).astype(int)
    df['is_weekend'] = (df['call_ts'].dt.dayofweek >= 5).astype(int)
    df['is_monday'] = (df['call_ts'].dt.dayofweek == 0).astype(int)
    df['is_friday'] = (df['call_ts'].dt.dayofweek == 4).astype(int)
    
    # Goa-specific seasonal features
    df['is_tourist_season'] = df['call_ts'].dt.month.isin([12, 1, 2, 3]).astype(int)
    df['is_monsoon'] = df['call_ts'].dt.month.isin([6, 7, 8, 9]).astype(int)
    df['is_festival_month'] = df['call_ts'].dt.month.isin([10, 11, 12, 1, 3, 4]).astype(int)
    
    return df

def create_lag_and_rolling_features(df, target_col='call_count'):
    """Create lag and rolling window features"""
    df = df.copy()
    df = df.sort_values('call_ts')
    
    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [3, 7, 14, 30]:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
    
    # Exponential weighted features
    df[f'{target_col}_ewm_3'] = df[target_col].ewm(span=3).mean()
    df[f'{target_col}_ewm_7'] = df[target_col].ewm(span=7).mean()
    
    return df

def create_interaction_features(df):
    """Create interaction features between important variables"""
    df = df.copy()
    
    # Time interactions
    df['hour_dow_interaction'] = df['call_ts'].dt.hour * df['call_ts'].dt.dayofweek
    df['hour_month_interaction'] = df['call_ts'].dt.hour * df['call_ts'].dt.month
    df['dow_month_interaction'] = df['call_ts'].dt.dayofweek * df['call_ts'].dt.month
    
    # Business logic interactions
    df['weekend_night'] = df['is_weekend'] * df['is_night']
    df['business_weekday'] = df['is_business_hour'] * (1 - df['is_weekend'])
    df['tourist_weekend'] = df['is_tourist_season'] * df['is_weekend']
    
    return df

def enhance_prophet_features(df):
    """Enhanced feature engineering for Prophet model"""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['call_ts'])
    
    # Aggregate by day for Prophet
    daily_df = df.groupby(df['ds'].dt.date).agg({
        'call_ts': 'count'
    }).reset_index()
    daily_df.columns = ['ds', 'y']
    daily_df['ds'] = pd.to_datetime(daily_df['ds'])
    
    # Add advanced features
    daily_df = create_advanced_temporal_features(daily_df)
    daily_df = create_lag_and_rolling_features(daily_df, 'y')
    
    return daily_df

def enhance_xgboost_features(df, festivals_list):
    """Enhanced feature engineering for XGBoost models"""
    df = df.copy()
    
    # Basic temporal features
    df = create_advanced_temporal_features(df)
    
    # Festival features with enhanced logic
    df['is_festival'] = 0
    df['festival_impact_score'] = 0
    df['days_to_festival'] = 999
    df['days_from_festival'] = 999
    
    festival_weights = {
        'New Year': 3.5, 'Christmas': 3.0, 'Diwali': 2.8, 'Holi': 2.5,
        'Carnival': 3.2, 'Shigmo': 2.7, 'Ganesh Chaturthi': 2.6
    }
    
    for name, start_date, end_date in festivals_list:
        mask = (df['call_ts'] >= start_date) & (df['call_ts'] <= end_date)
        df.loc[mask, 'is_festival'] = 1
        df.loc[mask, 'festival_impact_score'] = festival_weights.get(name, 1.0)
        
        # Days to/from festival
        for idx, row in df.iterrows():
            days_to = (pd.to_datetime(start_date) - row['call_ts']).days
            days_from = (row['call_ts'] - pd.to_datetime(end_date)).days
            
            if days_to >= 0 and days_to < df.loc[idx, 'days_to_festival']:
                df.loc[idx, 'days_to_festival'] = days_to
            if days_from >= 0 and days_from < df.loc[idx, 'days_from_festival']:
                df.loc[idx, 'days_from_festival'] = days_from
    
    # Geographic clustering features (simplified)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['geo_cluster'] = pd.cut(df['latitude'], bins=5, labels=False) * 10 + pd.cut(df['longitude'], bins=5, labels=False)
    else:
        df['geo_cluster'] = 0
    
    # Category encoding with frequency
    if 'category' in df.columns:
        category_freq = df['category'].value_counts()
        df['category_frequency'] = df['category'].map(category_freq)
        df['category_encoded'] = LabelEncoder().fit_transform(df['category'].fillna('unknown'))
    
    # Interaction features
    df = create_interaction_features(df)
    
    # Statistical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Limit to avoid too many features
        if col != 'call_ts':
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    return df

def select_best_features(X, y, k=50):
    """Select best features using statistical tests"""
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

def apply_feature_scaling(X_train, X_test=None):
    """Apply feature scaling for better model performance"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler