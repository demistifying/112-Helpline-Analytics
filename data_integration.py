# data_integration.py
import pandas as pd
import firebase_admin
from firebase_admin import firestore
from datetime import datetime
import uuid

def convert_caller_entry_to_dataset_format(caller_entry):
    """Convert caller entry from Firestore to dataset format"""
    
    # Generate call_id if not present
    call_id = caller_entry.get('event_id', f"CALL-{uuid.uuid4().hex[:8].upper()}")
    
    # Convert create_time to call_ts format
    create_time = caller_entry.get('create_time')
    if isinstance(create_time, str):
        try:
            call_ts = pd.to_datetime(create_time, dayfirst=True).strftime('%Y-%m-%d %H:%M:%S')
        except:
            # If parsing fails, skip this entry by returning None
            return None
    else:
        # If no create_time, skip this entry
        return None
    
    # Map event_main_type to category (simplified mapping)
    category_mapping = {
        'Crime': 'crime',
        'Accident': 'accident', 
        'Medical Emergency': 'medical',
        'Fire': 'fire',
        'Traffic': 'traffic',
        'Public Order': 'crime',
        'Domestic Dispute': 'crime',
        'Theft/Robbery': 'crime',
        'Missing Person': 'crime',
        'Noise Complaint': 'complaint',
        'Women Harassment': 'crime',
        'Child Harassment': 'crime',
        'Drink and Drove': 'traffic',
        'Drugs Associated': 'crime',
        'Fraud or cheating': 'crime',
        'Fighting and assault': 'crime',
        'Kidnapping': 'crime',
        'Murder': 'crime',
        'Property Offense': 'crime',
        'Robbery': 'crime',
        'Suicide': 'medical',
        'Terrorist attack': 'crime',
        'Threat': 'crime',
        'Other': 'other'
    }
    
    category = category_mapping.get(caller_entry.get('event_main_type', 'Other'), 'other')
    
    # Map station_main to jurisdiction
    jurisdiction = caller_entry.get('station_main', 'Unknown')
    
    # Default coordinates (Goa center) - in real implementation, you'd geocode the location
    caller_lat = 15.2993  # Default Goa latitude
    caller_lon = 74.1240  # Default Goa longitude
    
    return {
        'call_id': call_id,
        'call_ts': call_ts,
        'caller_lat': caller_lat,
        'caller_lon': caller_lon,
        'category': category,
        'jurisdiction': jurisdiction
    }

def get_new_caller_entries_as_dataframe():
    """Fetch new caller entries from Firestore and convert to DataFrame"""
    try:
        db = firestore.client()
        entries_ref = db.collection('caller_entries')
        entries = entries_ref.stream()
        
        converted_entries = []
        for entry in entries:
            entry_data = entry.to_dict()
            converted_data = convert_caller_entry_to_dataset_format(entry_data)
            if converted_data is not None:  # Only add valid entries
                converted_entries.append(converted_data)
        
        if converted_entries:
            return pd.DataFrame(converted_entries)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching caller entries: {e}")
        return pd.DataFrame()

def append_caller_entries_to_dataset(existing_df):
    """Append new caller entries to existing dataset"""
    new_entries_df = get_new_caller_entries_as_dataframe()
    
    if not new_entries_df.empty:
        # Remove duplicates based on call_id
        existing_call_ids = set(existing_df['call_id'].tolist()) if 'call_id' in existing_df.columns else set()
        new_entries_df = new_entries_df[~new_entries_df['call_id'].isin(existing_call_ids)]
        
        if not new_entries_df.empty:
            combined_df = pd.concat([existing_df, new_entries_df], ignore_index=True)
            return combined_df, len(new_entries_df)
    
    return existing_df, 0