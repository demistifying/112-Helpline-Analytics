import pandas as pd
import re
from difflib import get_close_matches
from collections import Counter

def clean_station_name(name):
    """Clean and normalize station names"""
    if pd.isna(name) or name == '':
        return name
    
    # Convert to string and strip whitespace
    name = str(name).strip()
    
    # Remove special characters and normalize
    name = re.sub(r'[;,\-_\s]+', ' ', name)  # Replace separators with space
    name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single space
    name = name.upper().strip()  # Convert to uppercase
    
    return name

def create_station_mapping(station_names):
    """Create mapping from variations to standardized names"""
    # Clean all names
    cleaned_names = [clean_station_name(name) for name in station_names if pd.notna(name)]
    
    # Count frequency of each cleaned name
    name_counts = Counter(cleaned_names)
    
    # Get most common names as standards
    standard_names = set()
    mapping = {}
    
    for name, count in name_counts.most_common():
        if name == '':
            continue
            
        # Find if this name is similar to any existing standard
        matches = get_close_matches(name, standard_names, n=1, cutoff=0.8)
        
        if matches:
            # Map to existing standard
            mapping[name] = matches[0]
        else:
            # This becomes a new standard
            standard_names.add(name)
            mapping[name] = name
    
    return mapping

def standardize_stations(df, station_columns):
    """Standardize station names in specified columns"""
    df_clean = df.copy()
    
    for col in station_columns:
        if col in df_clean.columns:
            print(f"Processing column: {col}")
            
            # Get all unique values
            unique_stations = df_clean[col].dropna().unique()
            print(f"Found {len(unique_stations)} unique values")
            
            # Create mapping
            mapping = create_station_mapping(unique_stations)
            
            # Apply mapping
            df_clean[col] = df_clean[col].apply(
                lambda x: mapping.get(clean_station_name(x), clean_station_name(x)) if pd.notna(x) else x
            )
            
            # Show standardization results
            new_unique = df_clean[col].dropna().unique()
            print(f"Reduced to {len(new_unique)} standardized values")
            print(f"Reduction: {len(unique_stations) - len(new_unique)} duplicates removed\n")
    
    return df_clean

def main():
    # Read the dataset
    print("Loading dataset...")
    df = pd.read_csv('data/Dummy_Dataset_Full.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Identify station-related columns
    station_columns = ['station_main', 'station_sub', 'incident_location']
    
    print("\nBefore standardization:")
    for col in station_columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            print(f"{col}: {unique_count} unique values")
    
    # Standardize station names
    print("\nStandardizing station names...")
    df_standardized = standardize_stations(df, station_columns)
    
    print("\nAfter standardization:")
    for col in station_columns:
        if col in df_standardized.columns:
            unique_count = df_standardized[col].nunique()
            print(f"{col}: {unique_count} unique values")
    
    # Save the cleaned dataset
    output_file = 'data/Dummy_Dataset_Full_Standardized.csv'
    df_standardized.to_csv(output_file, index=False)
    print(f"\nStandardized dataset saved as: {output_file}")
    
    # Show some examples of standardization
    print("\nSample standardization examples:")
    for col in station_columns:
        if col in df.columns:
            print(f"\n{col} examples:")
            original_values = df[col].dropna().unique()[:10]
            for orig in original_values:
                standardized = df_standardized[df[col] == orig][col].iloc[0] if len(df_standardized[df[col] == orig]) > 0 else orig
                if orig != standardized:
                    print(f"  '{orig}' -> '{standardized}'")

if __name__ == "__main__":
    main()