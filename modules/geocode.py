import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy import exc as geopy_exc
import logging
import time

logging.basicConfig(level=logging.INFO)

def map_locations_to_coords(df, location_col="incident_location", user_agent="goa-geocoder/1.0 (raghavshrivastav0812@gmail.com)"):
    """
    Map incident_location to real coordinates using OpenStreetMap (Nominatim) with caching.
    
    Parameters:
    - df: pandas.DataFrame
    - location_col: str, column name containing location descriptions
    - user_agent: str, user agent for Nominatim API (include contact email)
    
    Returns:
    - DataFrame with new columns: latitude, longitude
    """
    
    # Initialize geolocator — use a descriptive user_agent (include contact email per Nominatim policy)
    geolocator = Nominatim(user_agent=user_agent, timeout=10, scheme="https")
    # obey Nominatim rate limits and don't raise on transient errors (swallow_exceptions=True)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, swallow_exceptions=True)
    
    # Cache dictionary to avoid repeated lookups
    cache = {}
    
    # Storage for results
    lats, lons = [], []
    
    for loc in df[location_col]:
        if pd.isna(loc) or str(loc).strip() == "":
            lats.append(None)
            lons.append(None)
            continue
        
        # Normalize location string
        query = f"{loc}, Goa, India"
        
        # Check cache first
        if query in cache:
            lat, lon = cache[query]
        else:
            try:
                location = geocode(query)
                if location:
                    lat, lon = location.latitude, location.longitude
                else:
                    # geocode returned None (could be no match or swallowed exception like 403)
                    lat, lon = None, None
                cache[query] = (lat, lon)  # store in cache
            except geopy_exc.GeocoderInsufficientPrivileges as e:
                # 403 or insufficient privileges — log and continue without crashing
                logging.warning("GeocoderInsufficientPrivileges for query %s: %s", query, e)
                lat, lon = None, None
                cache[query] = (lat, lon)
            except Exception as e:
                logging.warning("Unexpected geocoding error for %s: %s", query, e)
                lat, lon = None, None
                cache[query] = (lat, lon)
                # small delay to avoid tight failure loop
                time.sleep(1)
        
        lats.append(lat)
        lons.append(lon)
    
    df["latitude"] = lats
    df["longitude"] = lons
    
    return df

# Load transformed dataset
df = pd.read_csv("transformed_112_calls.csv")

# Map to real coordinates
df_geo = map_locations_to_coords(df, location_col="incident_location")

# Save
df_geo.to_csv("Dummy_Dataset_Full.csv", index=False)