# modules/spike_detection.py
import pandas as pd
import numpy as np
from datetime import datetime

def detect_significant_days(df, category='crime', top_n=10):
    """
    Returns top festivals with highest call volume using ICS file data.
    Only includes festivals with significant spikes (>20% above baseline).
    """
    if df.empty:
        return []
    
    # Ensure proper date handling
    df = df.copy()
    if 'call_ts' in df.columns:
        df['date'] = pd.to_datetime(df['call_ts']).dt.date
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:
        return []
    
    # Filter for specific category
    df_cat = df[df['category'].str.lower() == category.lower()]
    if df_cat.empty:
        df_cat = df
    
    # Get daily call counts
    daily_counts = df_cat.groupby('date').size().reset_index()
    daily_counts.columns = ['date', 'count']
    
    if daily_counts.empty:
        return []
    
    # Calculate baseline (median is more robust than mean)
    baseline = daily_counts['count'].median()
    
    # Load festivals from ICS file
    from .festivals_ics import fetch_festivals_from_ics
    festivals_list = fetch_festivals_from_ics()
    
    # Filter out festivals not relevant to Goa
    goa_irrelevant = ['Karaka Chaturthi', 'Karva Chauth', 'Chhat Puja', 'Pratihar Sashthi', 
                      'Surya Sashthi', 'Guru Ravidas', 'Maharishi Dayanand', 'Valmiki Jayanti',
                      'Guru Govind Singh', 'Vasant Panchami', 'Holika Dahana', 'Gudi Padwa',
                      'Ugadi', 'Pongal', 'Makar Sankranti', 'Vaisakhi']
    
    # Create festival lookup by date (excluding irrelevant ones)
    festival_dates = {}
    for name, start_date, end_date in festivals_list:
        # Skip if festival name contains any irrelevant keyword
        if any(irrelevant.lower() in name.lower() for irrelevant in goa_irrelevant):
            continue
        
        # Handle date range
        current = start_date
        while current <= end_date:
            date_key = current.date()
            if date_key not in festival_dates:
                festival_dates[date_key] = name
            current = current + pd.Timedelta(days=1)
    
    # Find all festival days in data with significant spikes (>20% above baseline)
    festival_days = []
    for _, row in daily_counts.iterrows():
        date_obj = row['date']
        
        if date_obj in festival_dates:
            increase_pct = ((row['count'] - baseline) / baseline) * 100 if baseline > 0 else 0
            
            # Only include if there's a significant spike (>20% above baseline)
            if increase_pct > 20:
                festival_days.append({
                    'name': festival_dates[date_obj],
                    'max_day': pd.to_datetime(date_obj).strftime('%Y-%m-%d'),
                    'max_count': int(row['count']),
                    'baseline_avg': baseline,
                    'max_pct': increase_pct
                })
    
    # Deduplicate by festival name - keep only the highest count for each festival
    festival_dict = {}
    for day in festival_days:
        name = day['name']
        if name not in festival_dict or day['max_count'] > festival_dict[name]['max_count']:
            festival_dict[name] = day
    
    # Convert back to list and sort by call count
    festival_days = list(festival_dict.values())
    festival_days.sort(key=lambda x: x['max_count'], reverse=True)
    return festival_days[:top_n]

def create_spike_festivals(significant_days):
    """
    Convert significant days to festival-like format.
    """
    festivals = []
    for day in significant_days:
        date = pd.to_datetime(day['max_day'])
        festivals.append((day['name'], date, date))
    
    return festivals