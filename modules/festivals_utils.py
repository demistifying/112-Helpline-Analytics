# modules/festivals_utils.py
import pandas as pd

def filter_significant_festivals(
    festivals_in_range,
    df,
    category='crime',
    top_n=10
):
    """
    Identifies festivals with significant call volume increases (>30% above baseline).
    """
    if df.empty or not festivals_in_range:
        return []

    # Ensure proper date handling
    if 'call_ts' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['call_ts']).dt.date
    
    df['date'] = pd.to_datetime(df['date'])

    # Filter for the specific category
    df_cat = df[df['category'].str.lower() == category.lower()]
    if df_cat.empty:
        return []

    # Calculate overall baseline from all non-festival days
    all_festival_dates = set()
    for _, fs, fe in festivals_in_range:
        festival_days = pd.date_range(pd.to_datetime(fs).date(), pd.to_datetime(fe).date(), freq='D').date
        all_festival_dates.update(festival_days)
    
    non_festival_data = df_cat[~df_cat['date'].dt.date.isin(all_festival_dates)]
    if not non_festival_data.empty:
        baseline_avg = non_festival_data.groupby(non_festival_data['date'].dt.date).size().mean()
    else:
        baseline_avg = df_cat.groupby(df_cat['date'].dt.date).size().mean()
    
    if pd.isna(baseline_avg) or baseline_avg <= 0:
        baseline_avg = 1

    festival_crime_stats = []

    for name, fs, fe in festivals_in_range:
        festival_start = pd.to_datetime(fs).date()
        festival_end = pd.to_datetime(fe).date()
        festival_days = pd.date_range(festival_start, festival_end, freq='D').date
        
        df_fest = df_cat[df_cat['date'].dt.date.isin(festival_days)]
        if df_fest.empty:
            continue

        daily_counts = df_fest.groupby(df_fest['date'].dt.date).size()
        if daily_counts.empty:
            continue
            
        max_day_count = daily_counts.max()
        max_day = daily_counts.idxmax()
        increase_pct = ((max_day_count - baseline_avg) / baseline_avg) * 100

        # Include festivals with >30% increase OR top call volumes
        if increase_pct > 30 or max_day_count >= baseline_avg * 1.5:
            festival_crime_stats.append({
                'name': name,
                'max_day': max_day.strftime('%Y-%m-%d'),
                'max_count': int(max_day_count),
                'baseline_avg': baseline_avg,
                'max_pct': increase_pct
            })

    # Sort by percentage increase, then by max count
    sorted_festivals = sorted(festival_crime_stats, key=lambda x: (x['max_pct'], x['max_count']), reverse=True)
    return sorted_festivals[:top_n]