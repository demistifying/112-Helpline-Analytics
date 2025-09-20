# modules/mapping.py
# Placeholder mapping functions for Sprint-1.
# In Sprint-2 we'll replace / extend these to return Folium maps or GeoJSON.
import pydeck as pdk
import pandas as pd

def create_point_geojson(df, lat_col="caller_lat", lon_col="caller_lon", properties=None):
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


def clean_df_for_pydeck(df, lat_col="caller_lat", lon_col="caller_lon"):
    """Ensure DataFrame is JSON-serializable for Pydeck."""
    df = df[[lat_col, lon_col, "category", "jurisdiction"]].dropna().copy()

    # Force to float for coords
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce").astype(float)
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce").astype(float)

    # Ensure no NaNs
    df = df.dropna(subset=[lat_col, lon_col])

    # Convert other fields to string (for tooltip safety)
    df["category"] = df["category"].astype(str)
    df["jurisdiction"] = df["jurisdiction"].astype(str)

    df["weight"] = 1.0  # for heatmap intensity

    return df

def pydeck_points_map(df, lat_col="caller_lat", lon_col="caller_lon"):
    df = clean_df_for_pydeck(df, lat_col, lon_col)
    if df.empty:
        return None

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=[lon_col, lat_col],
        get_color=[0, 100, 255, 160],
        get_radius=80,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=df[lat_col].mean(),
        longitude=df[lon_col].mean(),
        zoom=9,
        pitch=0,
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state,
                    tooltip={"text": "{category}, {jurisdiction}"})

def pydeck_heatmap(df, lat_col="caller_lat", lon_col="caller_lon"):
    df = clean_df_for_pydeck(df, lat_col, lon_col)
    if df.empty:
        return None

    COLOR_RANGE = [
        [0, 0, 255, 25],    # blue
        [0, 255, 255, 85],  # cyan
        [0, 255, 0, 170],   # green
        [255, 255, 0, 200], # yellow
        [255, 0, 0, 255],   # red
    ]

    layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=[lon_col, lat_col],
        get_weight="weight",
        radiusPixels=40,   # reduce radius so clusters form
        intensity=2,
        threshold=0.05,     # filter very low density
        color_range = COLOR_RANGE
    )

    view_state = pdk.ViewState(
        latitude=df[lat_col].mean(),
        longitude=df[lon_col].mean(),
        zoom=9,
        pitch=0,
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state)