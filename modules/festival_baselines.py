# modules/festival_baselines.py
import pandas as pd

def calculate_weekly_top_n_peaks(df, date_col='date', value_col='count', n=2):
    """
    Calculates the average of the top N highest days for each week in the dataset.

    Args:
        df (pd.DataFrame): Daily aggregated call counts with date and count columns.
        date_col (str): The name of the date column.
        value_col (str): The name of the column with call counts.
        n (int): The number of top peaks to average per week.

    Returns:
        pd.DataFrame: A DataFrame with 'week_start' and 'avg_top_n_peak' columns.
    """
    if df.empty:
        return pd.DataFrame(columns=['week_start', f'avg_top_{n}_peak'])

    df[date_col] = pd.to_datetime(df[date_col])
    df['week_start'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')

    # Group by week and get the top N counts
    top_n_counts = df.groupby('week_start')[value_col].apply(
        lambda x: x.nlargest(n).mean()
    ).reset_index()

    top_n_counts.rename(columns={value_col: f'avg_top_{n}_peak'}, inplace=True)

    return top_n_counts

def get_baseline_for_date(target_date, baselines_df):
    """
    Finds the corresponding weekly baseline for a specific date.

    Args:
        target_date (pd.Timestamp): The date to find the baseline for.
        baselines_df (pd.DataFrame): The DataFrame with weekly baselines.

    Returns:
        float: The baseline value for that date's week. Returns global average if not found.
    """
    if baselines_df.empty:
        return 0.0 # Return a default value if no baselines are available

    target_week_start = target_date - pd.to_timedelta(target_date.dayofweek, unit='d')
    baseline_row = baselines_df[baselines_df['week_start'] == target_week_start]

    if not baseline_row.empty:
        return baseline_row.iloc[0, 1]  # The avg_top_n_peak value
    else:
        # Fallback to the most recent baseline if the exact week is not found
        if not baselines_df.empty:
            return baselines_df.iloc[-1, 1]
        return 0.0