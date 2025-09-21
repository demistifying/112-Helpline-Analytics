# modules/predictive_core.py
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ---------------- Festival Baseline Utilities ---------------- #
def avg_top2_weekly_peaks(df, date_col="date", value_col="calls"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["week"] = df[date_col].dt.isocalendar().week
    weekly_peaks = (
        df.groupby("week")[value_col]
        .apply(lambda x: x.nlargest(2).mean() if len(x) >= 2 else x.max())
    )
    return weekly_peaks.mean()

def flag_significant_festivals(festivals_list, df, threshold=0.3, date_col="date", value_col="calls"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    results = []
    for name, fs, fe in festivals_list:
        fs = pd.to_datetime(fs)   # ✅ convert to Timestamp
        fe = pd.to_datetime(fe)   # ✅ convert to Timestamp

        mask = (df[date_col] >= fs) & (df[date_col] <= fe)
        sub = df[mask]

        if not sub.empty:
            max_calls = sub[value_col].max()
            baseline = avg_top2_weekly_peaks(df, date_col, value_col)
            impact_pct = ((max_calls - baseline) / baseline) * 100
            if impact_pct >= threshold * 100:
                results.append({
                    "name": name,
                    "start": fs,
                    "end": fe,
                    "max_count": int(max_calls),
                    "impact_pct": impact_pct
                })
    return sorted(results, key=lambda x: x["impact_pct"], reverse=True)

# ---------------- Prophet Forecasting ---------------- #
def train_prophet(daily_df, holidays_df):
    dfp = daily_df.rename(columns={"date": "ds", "calls": "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    if not holidays_df.empty:
        m.add_country_holidays(country_name="IN")
        m = Prophet(holidays=holidays_df)
    m.fit(dfp)
    return m

def stepwise_forecast(history_df, n_days, holidays_df=None):
    """
    Perform stepwise daily forecasting using Prophet.
    history_df: DataFrame with columns ['ds', 'y']
    """
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    
    if holidays_df is not None and not holidays_df.empty:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    holidays=holidays_df)
    
    m.fit(history_df)

    preds = []
    hist = history_df.copy()

    for _ in range(n_days):
        future = pd.DataFrame({"ds": [hist["ds"].max() + pd.Timedelta(days=1)]})
        forecast = m.predict(future)
        yhat = forecast.loc[0, "yhat"]
        preds.append({"date": future["ds"].iloc[0], "predicted_calls": yhat})

        # extend history with predicted value
        hist = pd.concat([hist, pd.DataFrame({"ds": future["ds"], "y": [yhat]})])
        m.fit(hist)

    return pd.DataFrame(preds)

# ---------------- Event Type Classification ---------------- #
def train_event_model(df, event_col="event_type", festivals=None):
    df = df.copy()
    df["hour"] = pd.to_datetime(df["date"]).dt.hour
    df["weekday"] = pd.to_datetime(df["date"]).dt.weekday
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    if festivals:
        df["is_festival"] = df["date"].isin(festivals).astype(int)
    else:
        df["is_festival"] = 0

    X = df[["hour", "weekday", "month", "is_weekend", "is_festival"]]
    y = df[event_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(le.classes_),
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), target_names=le.classes_, output_dict=True)

    return model, le, report

def predict_event_distribution(model, le, future_dates, time_periods, festivals=None):
    results = []
    for date in future_dates:
        weekday = date.weekday()
        month = date.month
        is_weekend = int(weekday in [5, 6])
        is_festival = int(date in festivals) if festivals else 0

        for tp, hour in time_periods.items():
            x = pd.DataFrame([{
                "hour": hour, "weekday": weekday, "month": month,
                "is_weekend": is_weekend, "is_festival": is_festival
            }])
            probs = model.predict_proba(x)[0]
            for cls, p in zip(le.classes_, probs):
                results.append({"date": date, "time_period": tp, "event_type": cls, "probability": p})
    return pd.DataFrame(results)
