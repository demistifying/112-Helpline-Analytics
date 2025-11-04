# modules/feature_engineering.py
import pandas as pd
import numpy as np

def create_time_features(df, timestamp_col='call_ts'):
    """
    Creates comprehensive time-based features for high accuracy.
    """
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
    
    # Advanced time patterns
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
    
    # Cyclical encodings for better pattern recognition
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Time period categorization
    df['hour_category'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df = pd.get_dummies(df, columns=['hour_category'], prefix='time_period')
    
    return df

def create_festival_features(df, festivals_list, timestamp_col='call_ts'):
    """
    Creates enhanced festival features.
    """
    festival_dates = set()
    for _, start, end in festivals_list:
        current_date = start.date()
        while current_date <= end.date():
            festival_dates.add(current_date)
            current_date += pd.Timedelta(days=1)

    df['is_festival'] = df[timestamp_col].dt.date.isin(festival_dates).astype(int)
    
    # Festival proximity features
    df['days_to_festival'] = 999
    df['days_from_festival'] = 999
    
    for idx, row in df.iterrows():
        if pd.notna(row[timestamp_col]):
            current_date = row[timestamp_col].date()
            min_days_to = 999
            min_days_from = 999
            
            for festival_date in festival_dates:
                days_diff = (festival_date - current_date).days
                if days_diff >= 0 and days_diff < min_days_to:
                    min_days_to = days_diff
                elif days_diff < 0 and abs(days_diff) < min_days_from:
                    min_days_from = abs(days_diff)
            
            df.at[idx, 'days_to_festival'] = min_days_to if min_days_to < 999 else 0
            df.at[idx, 'days_from_festival'] = min_days_from if min_days_from < 999 else 0
    
    df['is_pre_festival'] = (df['days_to_festival'] <= 2).astype(int)
    df['is_post_festival'] = (df['days_from_festival'] <= 2).astype(int)
    
    return df

def prepare_features_for_prophet(df, significant_festivals=None):
    """
    Prepares enhanced DataFrame for Prophet with comprehensive features.
    """
    df_clean = df.dropna(subset=['call_ts']).copy()
    df_clean['call_ts'] = pd.to_datetime(df_clean['call_ts'])
    
    df_prophet = df_clean.set_index('call_ts').resample('D').size().reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Enhanced time features
    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['day_of_month'] = df_prophet['ds'].dt.day
    df_prophet['is_weekend'] = (df_prophet['day_of_week'] >= 5).astype(int)
    df_prophet['is_month_start'] = (df_prophet['day_of_month'] <= 5).astype(int)
    df_prophet['is_month_end'] = (df_prophet['day_of_month'] >= 25).astype(int)
    
    # Cyclical features
    df_prophet['day_sin'] = np.sin(2 * np.pi * df_prophet['day_of_week'] / 7)
    df_prophet['day_cos'] = np.cos(2 * np.pi * df_prophet['day_of_week'] / 7)
    df_prophet['month_sin'] = np.sin(2 * np.pi * df_prophet['month'] / 12)
    df_prophet['month_cos'] = np.cos(2 * np.pi * df_prophet['month'] / 12)
    
    # Advanced lag and rolling features
    df_prophet = df_prophet.sort_values('ds')
    df_prophet['lag_1'] = df_prophet['y'].shift(1).fillna(df_prophet['y'].mean())
    df_prophet['lag_3'] = df_prophet['y'].shift(3).fillna(df_prophet['y'].mean())
    df_prophet['lag_7'] = df_prophet['y'].shift(7).fillna(df_prophet['y'].mean())
    df_prophet['rolling_3'] = df_prophet['y'].rolling(3, min_periods=1).mean()
    df_prophet['rolling_7'] = df_prophet['y'].rolling(7, min_periods=1).mean()
    df_prophet['rolling_14'] = df_prophet['y'].rolling(14, min_periods=1).mean()
    
    # Statistical features
    df_prophet['y_std_7'] = df_prophet['y'].rolling(7, min_periods=1).std().fillna(0)
    df_prophet['y_trend'] = df_prophet['rolling_7'] - df_prophet['rolling_14']
    
    # Enhanced festival features
    df_prophet['is_festival'] = 0
    if significant_festivals:
        festival_dates = set()
        for fest_info in significant_festivals:
            if 'max_day' in fest_info:
                fest_date = pd.to_datetime(fest_info['max_day']).date()
                festival_dates.add(fest_date)
        
        df_prophet['is_festival'] = df_prophet['ds'].dt.date.isin(festival_dates).astype(int)
    
    return df_prophet

def prepare_features_for_xgboost(df, festivals_list):
    """
    Prepares comprehensive features for XGBoost models.
    """
    df_xgb = df.copy()
    if 'call_ts' not in df_xgb.columns:
        raise ValueError("'call_ts' column is required for feature engineering.")

    # Use comprehensive feature engineering
    df_xgb = create_advanced_features(df_xgb, festivals_list)

    if 'category' in df_xgb.columns:
        df_xgb['category'] = df_xgb['category'].astype('category')

    return df_xgb

def create_advanced_features(df, festivals_list):
    """
    Creates comprehensive feature set for maximum accuracy.
    """
    df_featured = df.copy()
    
    # Comprehensive time features
    df_featured = create_time_features(df_featured, 'call_ts')
    
    # Enhanced festival features
    if festivals_list:
        df_featured = create_festival_features(df_featured, festivals_list)
    else:
        df_featured['is_festival'] = 0
        df_featured['is_pre_festival'] = 0
        df_featured['is_post_festival'] = 0
    
    # Temporal aggregation features
    df_featured = df_featured.sort_values('call_ts')
    
    # Daily patterns
    df_featured['date'] = df_featured['call_ts'].dt.date
    daily_counts = df_featured.groupby('date').size().reset_index(name='daily_calls')
    daily_counts = daily_counts.sort_values('date')
    
    # Rolling averages and lags
    daily_counts['rolling_3d_avg'] = daily_counts['daily_calls'].rolling(3, min_periods=1).mean()
    daily_counts['rolling_7d_avg'] = daily_counts['daily_calls'].rolling(7, min_periods=1).mean()
    daily_counts['lag_1d'] = daily_counts['daily_calls'].shift(1).fillna(daily_counts['daily_calls'].mean())
    daily_counts['lag_7d'] = daily_counts['daily_calls'].shift(7).fillna(daily_counts['daily_calls'].mean())
    
    # Merge back
    df_featured = df_featured.merge(
        daily_counts[['date', 'rolling_3d_avg', 'rolling_7d_avg', 'lag_1d', 'lag_7d']], 
        on='date', how='left'
    )
    
    # Hourly and daily frequency features
    hourly_stats = df_featured.groupby('hour').size().reset_index(name='hour_frequency')
    daily_stats = df_featured.groupby('day_of_week').size().reset_index(name='dow_frequency')
    
    df_featured = df_featured.merge(hourly_stats, on='hour', how='left')
    df_featured = df_featured.merge(daily_stats, on='day_of_week', how='left')
    
    # Interaction features
    df_featured['weekend_night'] = df_featured['is_weekend'] * df_featured['is_night']
    df_featured['festival_weekend'] = df_featured['is_festival'] * df_featured['is_weekend']
    df_featured['rush_hour_weekday'] = df_featured['is_rush_hour'] * (1 - df_featured['is_weekend'])
    df_featured['business_hour_weekday'] = df_featured['is_business_hour'] * (1 - df_featured['is_weekend'])
    
    # Fill NaN values
    numeric_columns = df_featured.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df_featured.columns:
            df_featured[col] = df_featured[col].fillna(0)
    
    return df_featured