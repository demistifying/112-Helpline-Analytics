# modules/mapping.py
import pydeck as pdk
import pandas as pd
import numpy as np

def create_point_geojson(df, lat_col="latitude", lon_col="longitude", properties=None):
    """
    Create a simple GeoJSON FeatureCollection (dict) of points.
    properties: list of columns to include as properties for each feature
    """
    features = []
    if properties is None:
        properties = []

    for _, row in df.iterrows():
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            continue
        props = {p: (row.get(p) if p in row else None) for p in properties}
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": props
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}

def clean_df_for_pydeck(df, lat_col="latitude", lon_col="longitude"):
    """Ensure DataFrame is JSON-serializable for Pydeck with smart coordinate detection."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    
    required_cols = [lat_col, lon_col]
    available_cols = [col for col in required_cols if col in df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    # Add other columns if they exist
    for col in ["category", "jurisdiction", "station_sub", "incident_location"]:
        if col in df.columns:
            available_cols.append(col)
    
    df_clean = df[available_cols].dropna(subset=[lat_col, lon_col]).copy()
    
    if df_clean.empty:
        return df_clean
    
    # Force coordinates to float
    df_clean[lat_col] = pd.to_numeric(df_clean[lat_col], errors="coerce").astype(float)
    df_clean[lon_col] = pd.to_numeric(df_clean[lon_col], errors="coerce").astype(float)
    
    # Remove any remaining NaNs
    df_clean = df_clean.dropna(subset=[lat_col, lon_col])
    
    # Filter to Goa bounds for better visualization
    goa_lat_min, goa_lat_max = 14.53, 15.80
    goa_lon_min, goa_lon_max = 73.40, 74.20
    
    df_clean = df_clean[
        (df_clean[lat_col] >= goa_lat_min) & 
        (df_clean[lat_col] <= goa_lat_max) &
        (df_clean[lon_col] >= goa_lon_min) & 
        (df_clean[lon_col] <= goa_lon_max)
    ]
    
    # Convert categorical columns to string
    for col in ["category", "jurisdiction", "station_sub", "incident_location"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
    
    return df_clean

def pydeck_points_map(df, lat_col="latitude", lon_col="longitude"):
    """Simple points map using latitude/longitude columns"""
    if df is None or df.empty:
        return None
    
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
    
    # Get valid coordinates and location name if available
    cols_to_use = [lat_col, lon_col]
    if 'incident_location' in df.columns:
        cols_to_use.append('incident_location')
    
    df_coords = df[cols_to_use].dropna(subset=[lat_col, lon_col]).copy()
    if df_coords.empty:
        return None
    
    # Filter to Goa bounds
    df_coords = df_coords[
        (df_coords[lat_col] >= 14.53) & (df_coords[lat_col] <= 15.80) &
        (df_coords[lon_col] >= 73.40) & (df_coords[lon_col] <= 74.20)
    ]
    
    if df_coords.empty:
        return None
    
    # Aggregate by location with location name
    if 'incident_location' in df_coords.columns:
        agg_data = df_coords.groupby([lat_col, lon_col, 'incident_location']).size().reset_index(name='incident_count')
    else:
        agg_data = df_coords.groupby([lat_col, lon_col]).size().reset_index(name='incident_count')
        agg_data['incident_location'] = 'Unknown Location'
    
    agg_data['radius'] = 150 + (agg_data['incident_count'] / agg_data['incident_count'].max()) * 250
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=agg_data,
        get_position=[lon_col, lat_col],
        get_color=[255, 100, 100, 160],
        get_radius='radius',
        pickable=True,
    )
    
    view_state = pdk.ViewState(
        latitude=agg_data[lat_col].mean(),
        longitude=agg_data[lon_col].mean(),
        zoom=10,
        pitch=0,
    )
    
    return pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state,
        tooltip={"text": "{incident_location}\n{incident_count} incidents"}
    ), agg_data

def pydeck_heatmap(df, lat_col="latitude", lon_col="longitude"):
    """Create a heatmap using pydeck with smart coordinate detection"""
    if df is None or df.empty:
        return None
    
    # Smart column detection - find the column pair with the most valid data
    coord_options = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("Latitude", "Longitude")
    ]
    
    best_coords = None
    best_count = 0
    actual_lat_col = None
    actual_lon_col = None
    
    for lat_option, lon_option in coord_options:
        if lat_option in df.columns and lon_option in df.columns:
            test_coords = df[[lat_option, lon_option]].dropna()
            if not test_coords.empty:
                test_coords[lat_option] = pd.to_numeric(test_coords[lat_option], errors='coerce')
                test_coords[lon_option] = pd.to_numeric(test_coords[lon_option], errors='coerce')
                test_coords = test_coords.dropna()
                
                if len(test_coords) > best_count:
                    best_coords = test_coords
                    best_count = len(test_coords)
                    actual_lat_col = lat_option
                    actual_lon_col = lon_option
    
    if best_coords is not None:
        df_coords = best_coords
    else:
        df_coords = None
    
    if df_coords is None or df_coords.empty:
        return None
    
    # Filter to Goa bounds
    goa_lat_min, goa_lat_max = 14.53, 15.80
    goa_lon_min, goa_lon_max = 73.40, 74.20
    
    df_coords = df_coords[
        (df_coords[actual_lat_col] >= goa_lat_min) & 
        (df_coords[actual_lat_col] <= goa_lat_max) &
        (df_coords[actual_lon_col] >= goa_lon_min) & 
        (df_coords[actual_lon_col] <= goa_lon_max)
    ]
    
    if df_coords.empty:
        return None

    layer = pdk.Layer(
        "HeatmapLayer",
        data=df_coords,
        get_position=[actual_lon_col, actual_lat_col],
        get_weight=1,
        radius_pixels=60
    )

    view_state = pdk.ViewState(
        latitude=df_coords[actual_lat_col].mean(),
        longitude=df_coords[actual_lon_col].mean(),
        zoom=10,
        pitch=0
    )

    return pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state
    )

def pydeck_hexbin_3d(df, lat_col="latitude", lon_col="longitude"):
    """Create a spread heatmap with smooth gradient"""
    if df is None or df.empty:
        return None
    
    # Smart column detection
    coord_options = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("Latitude", "Longitude")
    ]
    
    best_coords = None
    best_count = 0
    actual_lat_col = None
    actual_lon_col = None
    
    for lat_option, lon_option in coord_options:
        if lat_option in df.columns and lon_option in df.columns:
            test_coords = df[[lat_option, lon_option]].dropna()
            if not test_coords.empty:
                test_coords[lat_option] = pd.to_numeric(test_coords[lat_option], errors='coerce')
                test_coords[lon_option] = pd.to_numeric(test_coords[lon_option], errors='coerce')
                test_coords = test_coords.dropna()
                
                if len(test_coords) > best_count:
                    best_coords = test_coords
                    best_count = len(test_coords)
                    actual_lat_col = lat_option
                    actual_lon_col = lon_option
    
    if best_coords is not None:
        df_coords = best_coords
    else:
        df_coords = None
    
    if df_coords is None or df_coords.empty:
        return None
    
    # Filter to Goa bounds
    goa_lat_min, goa_lat_max = 14.53, 15.80
    goa_lon_min, goa_lon_max = 73.40, 74.20
    
    df_coords = df_coords[
        (df_coords[actual_lat_col] >= goa_lat_min) & 
        (df_coords[actual_lat_col] <= goa_lat_max) &
        (df_coords[actual_lon_col] >= goa_lon_min) & 
        (df_coords[actual_lon_col] <= goa_lon_max)
    ]
    
    if df_coords.empty:
        return None
    
    layer = pdk.Layer(
        "HeatmapLayer",
        data=df_coords,
        get_position=[actual_lon_col, actual_lat_col],
        get_weight=1,
        radius_pixels=60,
        intensity=1.0,
        threshold=0.05,
        color_range=[
            [0, 0, 255, 255],
            [0, 128, 255, 255],
            [0, 255, 255, 255],
            [0, 255, 128, 255],
            [0, 255, 0, 255],
            [128, 255, 0, 255],
            [255, 255, 0, 255],
            [255, 200, 0, 255],
            [255, 128, 0, 255],
            [255, 64, 0, 255],
            [255, 0, 0, 255]
        ]
    )

    view_state = pdk.ViewState(
        latitude=df_coords[actual_lat_col].mean(),
        longitude=df_coords[actual_lon_col].mean(),
        zoom=10,
        pitch=0
    )

    return pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state
    )