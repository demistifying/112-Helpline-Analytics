# modules/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

def create_time_features(df, timestamp_col='call_ts'):
    """
    Creates comprehensive time-based features from a timestamp column.
    """
    # Basic Features
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    df['quarter'] = df[timestamp_col].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
    
    # Advanced Cyclical Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Time-based patterns
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
    df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 3)).astype(int)
    
    # Part of day with more granular categories
    def get_part_of_day(hour):
        if 5 <= hour < 8:
            return 'Early_Morning'
        elif 8 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 14:
            return 'Lunch'
        elif 14 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 20:
            return 'Evening'
        elif 20 <= hour < 23:
            return 'Night'
        else:
            return 'Late_Night'
    
    df['part_of_day'] = df['hour'].apply(get_part_of_day)
    df = pd.get_dummies(df, columns=['part_of_day'], prefix='pod', drop_first=False)
    
    # Season features
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Monsoon'
    
    df['season'] = df['month'].apply(get_season)
    df = pd.get_dummies(df, columns=['season'], prefix='season', drop_first=False)
    
    return df


def create_festival_features(df, festivals_list, timestamp_col='call_ts'):
    """
    Creates a binary flag for festival days.
    """
    festival_dates = set()
    for _, start, end in festivals_list:
        current_date = start.date()
        while current_date <= end.date():
            festival_dates.add(current_date)
            current_date += pd.Timedelta(days=1)

    df['is_festival'] = df[timestamp_col].dt.date.isin(festival_dates).astype(int)
    return df

def prepare_features_for_prophet(df, significant_festivals=None):
    """
    Prepares enhanced DataFrame for Prophet with festival-aware features.
    """
    # Ensure call_ts is datetime and remove any NaT values
    df_clean = df.dropna(subset=['call_ts']).copy()
    df_clean['call_ts'] = pd.to_datetime(df_clean['call_ts'])
    
    df_prophet = df_clean.set_index('call_ts').resample('D').size().reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Add key regressors
    df_prophet['is_weekend'] = (df_prophet['ds'].dt.dayofweek >= 5).astype(int)
    df_prophet['month'] = df_prophet['ds'].dt.month
    
    # Add lag features for better prediction
    df_prophet = df_prophet.sort_values('ds')
    df_prophet['y_lag1'] = df_prophet['y'].shift(1).fillna(df_prophet['y'].mean())
    df_prophet['y_lag7'] = df_prophet['y'].shift(7).fillna(df_prophet['y'].mean())
    
    # Add rolling averages
    df_prophet['y_roll3'] = df_prophet['y'].rolling(window=3, min_periods=1).mean()
    df_prophet['y_roll7'] = df_prophet['y'].rolling(window=7, min_periods=1).mean()
    
    # Enhanced festival features
    df_prophet['is_festival'] = 0
    
    # Use significant festivals if provided
    if significant_festivals:
        festival_dates = set()
        for fest_info in significant_festivals:
            if 'max_day' in fest_info:
                fest_date = pd.to_datetime(fest_info['max_day']).date()
                festival_dates.add(fest_date)
        
        df_prophet['is_festival'] = df_prophet['ds'].dt.date.isin(festival_dates).astype(int)
    elif 'is_festival' in df.columns:
        # Fallback to original festival detection
        festival_daily = df.groupby(df['call_ts'].dt.date)['is_festival'].max().reset_index()
        festival_daily['call_ts'] = pd.to_datetime(festival_daily['call_ts'])
        df_prophet = df_prophet.merge(festival_daily, left_on='ds', right_on='call_ts', how='left', suffixes=('', '_fest'))
        df_prophet['is_festival'] = df_prophet['is_festival_fest'].fillna(0)
        df_prophet.drop(['call_ts', 'is_festival_fest'], axis=1, inplace=True)
    
    return df_prophet

def prepare_features_for_xgboost(df, festivals_list):
    """
    Prepares comprehensive features for XGBoost models using advanced feature engineering.
    """
    df_xgb = df.copy()
    if 'call_ts' not in df_xgb.columns:
        raise ValueError("'call_ts' column is required for feature engineering.")

    # Use the advanced feature engineering
    df_xgb = create_advanced_features(df_xgb, festivals_list)

    if 'category' in df_xgb.columns:
        df_xgb['category'] = df_xgb['category'].astype('category')

    return df_xgb

def add_lag_and_rolling_features(df, lag_hours=[1], rolling_windows=[3]):
    """
    Adds lag and rolling average features to the DataFrame.
    """
    df = df.sort_values('call_ts')
    df.set_index('call_ts', inplace=True)
    
    # Create lag features
    for lag in lag_hours:
        df[f'lag_{lag}hr_calls'] = df['category'].shift(lag).rolling(window=lag).count()
    
    # Create rolling average features
    for window in rolling_windows:
        df[f'rolling_{window}hr_avg_calls'] = df['category'].shift(1).rolling(window=window).count() / window
    
    df.reset_index(inplace=True)
    return df

def create_advanced_features(df, festivals_list):
    """
    Creates comprehensive feature set for maximum model performance.
    """
    df_featured = df.copy()
    
    # Enhanced time features
    df_featured = create_time_features(df_featured, 'call_ts')
    
    # Festival features with proximity
    festival_dates = set()
    for _, start, end in festivals_list:
        current_date = start.date()
        while current_date <= end.date():
            festival_dates.add(current_date)
            current_date += pd.Timedelta(days=1)
    
    df_featured['is_festival'] = df_featured['call_ts'].dt.date.isin(festival_dates).astype(int)
    
    # Festival proximity features
    df_featured['days_to_festival'] = 999
    df_featured['days_from_festival'] = 999
    
    # Filter out rows with NaT values in call_ts
    valid_rows = df_featured['call_ts'].notna()
    
    for idx, row in df_featured[valid_rows].iterrows():
        try:
            current_date = row['call_ts'].date()
            min_days_to = 999
            min_days_from = 999
            
            for festival_date in festival_dates:
                days_diff = (festival_date - current_date).days
                if days_diff >= 0 and days_diff < min_days_to:
                    min_days_to = days_diff
                elif days_diff < 0 and abs(days_diff) < min_days_from:
                    min_days_from = abs(days_diff)
            
            df_featured.at[idx, 'days_to_festival'] = min_days_to if min_days_to < 999 else 0
            df_featured.at[idx, 'days_from_festival'] = min_days_from if min_days_from < 999 else 0
        except (AttributeError, TypeError):
            # Skip rows with invalid dates
            continue
    
    df_featured['is_pre_festival'] = (df_featured['days_to_festival'] <= 3).astype(int)
    df_featured['is_post_festival'] = (df_featured['days_from_festival'] <= 3).astype(int)
    
    # Location-based features
    if 'caller_lat' in df_featured.columns and 'caller_lon' in df_featured.columns:
        # Geographic clustering
        coords = df_featured[['caller_lat', 'caller_lon']].dropna()
        if len(coords) > 10:
            kmeans = KMeans(n_clusters=min(10, len(coords)//10), random_state=42)
            df_featured['geo_cluster'] = -1
            df_featured.loc[coords.index, 'geo_cluster'] = kmeans.fit_predict(coords)
            df_featured = pd.get_dummies(df_featured, columns=['geo_cluster'], prefix='geo', drop_first=False)
    
    # Response time features
    if 'response_time_min' in df_featured.columns:
        df_featured['response_time_min'] = pd.to_numeric(df_featured['response_time_min'], errors='coerce')
        df_featured['is_quick_response'] = (df_featured['response_time_min'] <= 5).astype(int)
        df_featured['is_slow_response'] = (df_featured['response_time_min'] >= 15).astype(int)
        df_featured['response_time_log'] = np.log1p(df_featured['response_time_min'].fillna(0))
    
    # Jurisdiction features
    if 'jurisdiction' in df_featured.columns:
        agg_dict = {'call_ts': 'count'}
        if 'response_time_min' in df_featured.columns:
            agg_dict['response_time_min'] = ['mean', 'std']
            
        jurisdiction_stats = df_featured.groupby('jurisdiction').agg(agg_dict).fillna(0)
        
        if 'response_time_min' in df_featured.columns:
            jurisdiction_stats.columns = ['juri_call_count', 'juri_avg_response', 'juri_std_response']
        else:
            jurisdiction_stats.columns = ['juri_call_count']
            
        df_featured = df_featured.merge(jurisdiction_stats, left_on='jurisdiction', right_index=True, how='left')
        df_featured = pd.get_dummies(df_featured, columns=['jurisdiction'], prefix='juri', drop_first=False)
    
    # Temporal aggregation features
    df_featured = df_featured.sort_values('call_ts')
    
    # Hourly patterns
    hourly_stats = df_featured.groupby('hour').agg({
        'call_ts': 'count'
    }).rename(columns={'call_ts': 'hour_call_frequency'})
    df_featured = df_featured.merge(hourly_stats, left_on='hour', right_index=True, how='left')
    
    # Daily patterns
    daily_stats = df_featured.groupby('day_of_week').agg({
        'call_ts': 'count'
    }).rename(columns={'call_ts': 'dow_call_frequency'})
    df_featured = df_featured.merge(daily_stats, left_on='day_of_week', right_index=True, how='left')
    
    # Rolling window features
    df_featured['date'] = pd.to_datetime(df_featured['call_ts'].dt.date)
    daily_counts = df_featured.groupby('date').size().reset_index(name='daily_calls')
    daily_counts = daily_counts.sort_values('date')
    
    # Rolling averages
    daily_counts['rolling_3d_avg'] = daily_counts['daily_calls'].rolling(window=3, min_periods=1).mean()
    daily_counts['rolling_7d_avg'] = daily_counts['daily_calls'].rolling(window=7, min_periods=1).mean()
    daily_counts['rolling_14d_avg'] = daily_counts['daily_calls'].rolling(window=14, min_periods=1).mean()
    
    # Lag features
    daily_counts['lag_1d'] = daily_counts['daily_calls'].shift(1)
    daily_counts['lag_7d'] = daily_counts['daily_calls'].shift(7)
    
    # Merge back
    df_featured = df_featured.merge(daily_counts[['date', 'rolling_3d_avg', 'rolling_7d_avg', 'rolling_14d_avg', 'lag_1d', 'lag_7d']], 
                                   on='date', how='left')
    
    # Interaction features
    df_featured['weekend_night'] = df_featured['is_weekend'] * df_featured['is_night']
    df_featured['festival_weekend'] = df_featured['is_festival'] * df_featured['is_weekend']
    df_featured['rush_hour_weekday'] = df_featured['is_rush_hour'] * (1 - df_featured['is_weekend'])
    df_featured['business_hour_weekday'] = df_featured['is_business_hour'] * (1 - df_featured['is_weekend'])
    
    # Category encoding if present
    if 'category' in df_featured.columns:
        agg_dict = {'call_ts': 'count'}
        if 'response_time_min' in df_featured.columns:
            agg_dict['response_time_min'] = 'mean'
            
        category_stats = df_featured.groupby('category').agg(agg_dict).fillna(0)
        
        if 'response_time_min' in df_featured.columns:
            category_stats.columns = ['cat_frequency', 'cat_avg_response']
        else:
            category_stats.columns = ['cat_frequency']
            
        df_featured = df_featured.merge(category_stats, left_on='category', right_index=True, how='left')
    
    # Fill NaN values safely
    numeric_columns = df_featured.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df_featured.columns:
            df_featured[col] = df_featured[col].fillna(0)
    
    return df_featured


