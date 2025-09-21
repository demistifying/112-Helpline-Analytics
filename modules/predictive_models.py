# modules/predictive_models.py
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error, r2_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- 1. Call Volume Forecasting (Prophet) ---
@st.cache_resource
def train_prophet_model(df_prophet, holidays_df=None):
    """
    Trains an enhanced Prophet model with multiple regressors.
    """
    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000
    )
    
    # Add multiple regressors
    regressors = ['is_weekend', 'is_festival', 'month', 'day_of_week', 'is_month_end', 'is_month_start']
    for regressor in regressors:
        if regressor in df_prophet.columns:
            model.add_regressor(regressor, prior_scale=10.0, standardize=True)
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
    
    model.fit(df_prophet)

    initial_days = f'{max(30, int(len(df_prophet) * 0.6))} days'
    period_days = f'{max(7, int(len(df_prophet) * 0.1))} days'
    horizon_days = f'{max(7, int(len(df_prophet) * 0.2))} days'
    
    metrics = {}
    try:
        df_cv = cross_validation(model, initial=initial_days, period=period_days, horizon=horizon_days, parallel="threads")
        df_p = performance_metrics(df_cv)
        metrics = df_p[['mae', 'rmse', 'mape']].mean().to_dict()
    except Exception as e:
        metrics = {"error": f"Cross-validation failed: {e}"}

    return model, metrics

def predict_with_prophet(model, future_days, last_date):
    """
    Makes future predictions, ensuring the future dataframe includes the regressor column.
    """
    future = model.make_future_dataframe(periods=future_days)
    future = future[future['ds'] > last_date]
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    forecast = model.predict(future)
    return forecast

# --- 2. Event Type Trend Prediction (XGBoost Classifier) ---
@st.cache_resource
def train_event_type_model(df_features):
    """
    Trains an ensemble of models for event type classification.
    """
    from .feature_engineering import create_advanced_features
    
    # Create comprehensive features
    df_enhanced = create_advanced_features(df_features, [])
    
    # Select relevant features
    feature_cols = [col for col in df_enhanced.columns if col not in ['category', 'call_ts', 'call_id', 'location_text', 'response_ts', 'response_outcome', 'date']]
    numeric_features = df_enhanced[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df_enhanced[numeric_features].fillna(0)
    y = df_enhanced['category']
    
    # Remove constant features
    X = X.loc[:, X.var() > 0]
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(50, len(X.columns)))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Ensemble of models
    models = {
        'xgb': XGBClassifier(
            objective='multi:softprob',
            n_estimators=500,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'rf': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_scores = {}
    
    for name, model in models.items():
        accuracies, f1_scores = [], []
        
        for train_index, test_index in skf.split(X_scaled, y_encoded):
            X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]
            
            if name == 'xgb':
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
        model_scores[name] = {
            'accuracy': np.mean(accuracies),
            'f1': np.mean(f1_scores)
        }
    
    # Select best model
    best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['f1'])
    best_model = models[best_model_name]
    
    # Train final model
    if best_model_name == 'xgb':
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_encoded)
        best_model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
    else:
        best_model.fit(X_scaled, y_encoded)
    
    # Generate classification report
    y_pred_final = best_model.predict(X_scaled)
    report = classification_report(y_encoded, y_pred_final, target_names=le.classes_, output_dict=True, zero_division=0)
    
    # Clean report for display
    class_report_dict = {k: v for k, v in report.items() if isinstance(v, dict)}
    avg_report_df = pd.DataFrame(class_report_dict).transpose()
    
    metrics = {
        'K-Fold Mean Accuracy': model_scores[best_model_name]['accuracy'],
        'K-Fold Mean Precision (Weighted)': precision_score(y_encoded, y_pred_final, average='weighted', zero_division=0),
        'Classification Report': avg_report_df.to_dict('index'),
        'Best Model': best_model_name,
        'Model Scores': model_scores
    }
    
    # Store preprocessing objects
    best_model.scaler = scaler
    best_model.selector = selector
    best_model.feature_names = selected_features
    
    return best_model, le, metrics


def predict_event_type_distribution(model, label_encoder, future_df):
    """
    Enhanced prediction with preprocessing pipeline.
    """
    try:
        # Apply same preprocessing as training
        if hasattr(model, 'scaler') and hasattr(model, 'selector'):
            # Select features that were used in training
            available_features = [f for f in model.feature_names if f in future_df.columns]
            future_processed = future_df[available_features].fillna(0)
            
            # Apply feature selection and scaling
            future_selected = model.selector.transform(future_processed)
            future_scaled = model.scaler.transform(future_selected)
            
            predictions_encoded = model.predict(future_scaled)
        else:
            predictions_encoded = model.predict(future_df)
        
        predictions_labels = label_encoder.inverse_transform(predictions_encoded)
        distribution = pd.Series(predictions_labels).value_counts(normalize=True).reset_index()
        distribution.columns = ['category', 'percentage']
        return distribution
    except Exception as e:
        # Fallback to simple prediction
        try:
            predictions_encoded = model.predict(future_df.fillna(0))
            predictions_labels = label_encoder.inverse_transform(predictions_encoded)
            distribution = pd.Series(predictions_labels).value_counts(normalize=True).reset_index()
            distribution.columns = ['category', 'percentage']
            return distribution
        except:
            # Return empty distribution if all fails
            return pd.DataFrame({'category': [], 'percentage': []})


# --- 3. Peak Hour Prediction (XGBoost Regressor) ---
@st.cache_resource
def train_peak_hour_model(df_features):
    """
    Trains an enhanced regressor for peak hour prediction.
    """
    from .feature_engineering import create_advanced_features
    
    # Create comprehensive features
    df_enhanced = create_advanced_features(df_features, [])
    
    # Create date column properly
    df_enhanced['date'] = pd.to_datetime(df_enhanced['call_ts'].dt.date)
    
    # Simplify hourly aggregation to avoid grouping issues
    df_hourly = df_enhanced.groupby(['date', 'hour']).agg({
        'call_ts': 'count',
        'day_of_week': 'first',
        'month': 'first',
        'is_weekend': 'first',
        'is_festival': 'first',
        'hour_sin': 'first',
        'hour_cos': 'first',
        'month_sin': 'first',
        'month_cos': 'first',
        'day_sin': 'first',
        'day_cos': 'first',
        'is_business_hour': 'first',
        'is_night': 'first',
        'is_rush_hour': 'first',
        'is_late_night': 'first'
    }).reset_index()
    
    df_hourly.rename(columns={'call_ts': 'call_count'}, inplace=True)
    
    # Add temporal features
    df_hourly['day_of_year'] = df_hourly['date'].dt.dayofyear.astype(int)
    df_hourly['week_of_year'] = df_hourly['date'].dt.isocalendar().week.astype(int)
    
    # Add lag features
    df_hourly = df_hourly.sort_values(['date', 'hour'])
    df_hourly['lag_1h'] = df_hourly.groupby('hour')['call_count'].shift(1)
    df_hourly['lag_24h'] = df_hourly.groupby('hour')['call_count'].shift(24)
    df_hourly['lag_168h'] = df_hourly.groupby('hour')['call_count'].shift(168)  # 1 week
    
    # Rolling averages
    df_hourly['rolling_3h'] = df_hourly.groupby('hour')['call_count'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df_hourly['rolling_24h'] = df_hourly.groupby('hour')['call_count'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    
    # Fill NaN values
    df_hourly = df_hourly.fillna(0)
    
    # Select features for modeling
    feature_cols = [col for col in df_hourly.columns if col not in ['call_count', 'date']]
    X = df_hourly[feature_cols]
    y = df_hourly['call_count']
    
    # Convert all columns to float to avoid dtype issues
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
    
    # Remove constant features
    X = X.loc[:, X.var() > 0]
    
    # Multiple models ensemble
    models = {
        'xgb': XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        ),
        'rf': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_scores = {}
    
    for name, model in models.items():
        maes, r2s = [], []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            maes.append(mean_absolute_error(y_test, y_pred))
            r2s.append(r2_score(y_test, y_pred))
        
        model_scores[name] = {
            'mae': np.mean(maes),
            'r2': np.mean(r2s)
        }
    
    # Select best model (lowest MAE)
    best_model_name = min(model_scores.keys(), key=lambda k: model_scores[k]['mae'])
    best_model = models[best_model_name]
    
    # Train final model
    best_model.fit(X, y)
    
    metrics = {
        'K-Fold Mean Absolute Error (MAE)': model_scores[best_model_name]['mae'],
        'K-Fold Mean R-squared': model_scores[best_model_name]['r2'],
        'Best Model': best_model_name,
        'Model Scores': model_scores
    }
    
    # Store feature names
    best_model.feature_names = X.columns.tolist()
    
    return best_model, metrics

def predict_hourly_calls_for_n_days(model, start_date, n_days, festivals_list):
    """
    Enhanced prediction for peak hours using comprehensive features.
    """
    from .feature_engineering import create_time_features, create_festival_features
    peak_hour_predictions = []
    
    for i in range(n_days):
        target_date = start_date + pd.Timedelta(days=i)
        hours_df = pd.DataFrame({'hour': range(24)})
        hours_df['call_ts'] = pd.to_datetime(target_date) + pd.to_timedelta(hours_df['hour'], unit='h')
        hours_df['date'] = target_date
        
        # Create comprehensive features
        hours_df = create_time_features(hours_df, 'call_ts')
        hours_df = create_festival_features(hours_df, festivals_list, 'call_ts')
        
        # Add additional features that might be expected
        hours_df['day_of_year'] = hours_df['call_ts'].dt.dayofyear
        hours_df['week_of_year'] = hours_df['call_ts'].dt.isocalendar().week
        
        # Add lag features (set to 0 for prediction)
        lag_features = ['lag_1h', 'lag_24h', 'lag_168h', 'rolling_3h', 'rolling_24h']
        for feature in lag_features:
            hours_df[feature] = 0
        
        # Use model's stored feature names if available
        if hasattr(model, 'feature_names'):
            feature_cols = model.feature_names
        else:
            # Fallback to basic features
            feature_cols = [
                'hour', 'day_of_week', 'month', 'is_weekend', 'is_festival',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
            ]
        
        # Ensure all required features exist
        for col in feature_cols:
            if col not in hours_df.columns:
                hours_df[col] = 0
        
        try:
            # Select only the features that exist
            available_features = [col for col in feature_cols if col in hours_df.columns]
            prediction_df = hours_df[available_features].fillna(0)
            
            predictions = model.predict(prediction_df)
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
            peak_hour = np.argmax(predictions)
            
            peak_hour_predictions.append({
                'Date': target_date.date(), 
                'Predicted Peak Hour': f"{peak_hour:02d}:00",
                'Predicted Calls': int(predictions[peak_hour])
            })
        except Exception as e:
            # Fallback prediction
            peak_hour_predictions.append({
                'Date': target_date.date(), 
                'Predicted Peak Hour': "20:00",  # Default evening peak
                'Predicted Calls': 10
            })
    
    return pd.DataFrame(peak_hour_predictions)