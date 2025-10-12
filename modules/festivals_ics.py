# modules/festivals_ics.py (Fixed version with encoding handling)
import streamlit as st
import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import os

# Cache for 1 hour to avoid repeated downloads
@st.cache_data(ttl=3600)
def fetch_festivals_from_ics():
    """
    Fetch festival data from ICS file (local or remote) with proper encoding handling.
    Returns list of tuples: (name, start_date, end_date)
    """
    festivals = []
    
    # Try both local files
    local_files = [
        os.path.join("data", "festivals.ics"),
        os.path.join("data", "indian_festivals.ics")
    ]
    
    all_festivals = []
    for local_ics_path in local_files:
        if os.path.exists(local_ics_path):
            try:
                file_festivals = parse_local_ics_file(local_ics_path)
                if file_festivals:
                    all_festivals.extend(file_festivals)
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error parsing {local_ics_path}: {str(e)}")
    
    if all_festivals:
        # Remove duplicates and add New Year's Eve if missing
        all_festivals = list(set(all_festivals))
        
        # Check if New Year's Eve exists, if not add it
        current_year = datetime.now().year
        nye_exists = any('New Year' in name and '31' in str(date.day) for name, date, _ in all_festivals)
        if not nye_exists:
            for year in [current_year - 1, current_year, current_year + 1]:
                all_festivals.append((f"New Year's Eve", datetime(year, 12, 31), datetime(year, 12, 31)))
        
        st.sidebar.success(f"✅ Loaded {len(all_festivals)} festivals from local files")
        return all_festivals
    
    # If local fails, try remote sources
    remote_urls = [
        # Indian government holiday calendar
        "https://www.calendarlabs.com/ical-calendar/india-holidays-42.ics",
        # Alternative sources
        "https://calendar.google.com/calendar/ical/en.indian%23holiday%40group.v.calendar.google.com/public/basic.ics"
    ]
    
    for url in remote_urls:
        try:
            festivals = fetch_from_remote_ics(url)
            if festivals:
                st.sidebar.success(f"✅ Loaded {len(festivals)} festivals from remote source")
                return festivals
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to fetch from {url[:50]}...")
            continue
    
    # If all fails, return hardcoded major festivals
    st.sidebar.info("📅 Using fallback festival data")
    return get_fallback_festivals()

def parse_local_ics_file(file_path):
    """Parse local ICS file with multiple encoding attempts"""
    festivals = []
    
    # Try different encodings in order of preference
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                content = file.read()
                festivals = parse_ics_content(content)
                if festivals:  # If we got valid data, break
                    print(f"Successfully parsed ICS file with encoding: {encoding}")
                    break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {str(e)}")
            continue
    
    return festivals

def fetch_from_remote_ics(url):
    """Fetch and parse ICS from remote URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Try to decode with proper encoding
        content = response.content.decode('utf-8', errors='ignore')
        return parse_ics_content(content)
        
    except Exception as e:
        print(f"Error fetching from {url}: {str(e)}")
        return []

def parse_ics_content(content):
    """Parse ICS content and extract festival information"""
    festivals = []
    
    try:
        # Split content into events
        events = re.findall(r'BEGIN:VEVENT(.*?)END:VEVENT', content, re.DOTALL | re.IGNORECASE)
        
        for event in events:
            try:
                # Extract event details
                summary_match = re.search(r'SUMMARY:(.*)', event, re.IGNORECASE)
                dtstart_match = re.search(r'DTSTART[^:]*:(\d{8})', event, re.IGNORECASE)
                dtend_match = re.search(r'DTEND[^:]*:(\d{8})', event, re.IGNORECASE)
                
                if summary_match and dtstart_match:
                    name = summary_match.group(1).strip()
                    # Clean up the name
                    name = re.sub(r'\\[ntr]', ' ', name)  # Remove escaped characters
                    name = re.sub(r'[^\w\s-]', '', name)  # Remove special chars except hyphens
                    name = ' '.join(name.split())  # Normalize whitespace
                    
                    if not name or len(name) < 3:  # Skip very short or empty names
                        continue
                    
                    start_str = dtstart_match.group(1)
                    end_str = dtend_match.group(1) if dtend_match else start_str
                    
                    # Parse dates
                    start_date = datetime.strptime(start_str, '%Y%m%d')
                    end_date = datetime.strptime(end_str, '%Y%m%d')
                    
                    # Only include festivals within a reasonable range (past 1 year to future 2 years)
                    current_year = datetime.now().year
                    if (current_year - 1) <= start_date.year <= (current_year + 2):
                        festivals.append((name, start_date, end_date))
                        
            except Exception as e:
                # Skip individual events that fail to parse
                continue
    
    except Exception as e:
        print(f"Error parsing ICS content: {str(e)}")
        return []
    
    # Remove duplicates and sort by date
    festivals = list(set(festivals))  # Remove duplicates
    festivals.sort(key=lambda x: x[1])  # Sort by start date
    
    return festivals

def get_fallback_festivals():
    """Return hardcoded major Indian festivals as fallback"""
    current_year = datetime.now().year
    
    # Major Indian festivals (approximate dates - you should update with exact dates)
    festivals = []
    
    # Add festivals for current and next year
    # Add festivals for past, current and next year to ensure coverage
    for year in [current_year - 1, current_year, current_year + 1]:
        festivals.extend([
            (f"New Year's Day {year}", datetime(year, 1, 1), datetime(year, 1, 1)),
            (f"Republic Day {year}", datetime(year, 1, 26), datetime(year, 1, 26)),
            (f"Maha Shivratri {year}", datetime(year, 2, 18), datetime(year, 2, 18)),
            (f"Holi {year}", datetime(year, 3, 13), datetime(year, 3, 13)),
            (f"Ram Navami {year}", datetime(year, 4, 17), datetime(year, 4, 17)),
            (f"Good Friday {year}", datetime(year, 4, 7), datetime(year, 4, 7)),
            (f"Eid ul-Fitr {year}", datetime(year, 4, 21), datetime(year, 4, 21)),
            (f"Buddha Purnima {year}", datetime(year, 5, 16), datetime(year, 5, 16)),
            (f"Eid ul-Adha {year}", datetime(year, 6, 28), datetime(year, 6, 28)),
            (f"Independence Day {year}", datetime(year, 8, 15), datetime(year, 8, 15)),
            (f"Janmashtami {year}", datetime(year, 8, 30), datetime(year, 8, 30)),
            (f"Ganesh Chaturthi {year}", datetime(year, 9, 7), datetime(year, 9, 7)),
            (f"Gandhi Jayanti {year}", datetime(year, 10, 2), datetime(year, 10, 2)),
            (f"Dussehra {year}", datetime(year, 10, 15), datetime(year, 10, 15)),
            (f"Karva Chauth {year}", datetime(year, 11, 1), datetime(year, 11, 1)),
            (f"Diwali {year}", datetime(year, 11, 12), datetime(year, 11, 12)),
            (f"Bhai Dooj {year}", datetime(year, 11, 14), datetime(year, 11, 14)),
            (f"Guru Nanak Jayanti {year}", datetime(year, 11, 27), datetime(year, 11, 27)),
            (f"Christmas {year}", datetime(year, 12, 25), datetime(year, 12, 25)),
            (f"New Year's Eve", datetime(year, 12, 31), datetime(year, 12, 31)),
        ])
    
    return festivals

def create_sample_ics_file():
    """Create a sample ICS file for testing (in case the original is corrupted)"""
    sample_content = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//Test//EN
BEGIN:VEVENT
SUMMARY:Republic Day
DTSTART:20250126
DTEND:20250126
DESCRIPTION:India's Republic Day
END:VEVENT
BEGIN:VEVENT
SUMMARY:Independence Day
DTSTART:20250815
DTEND:20250815
DESCRIPTION:India's Independence Day
END:VEVENT
BEGIN:VEVENT
SUMMARY:Gandhi Jayanti
DTSTART:20251002
DTEND:20251002
DESCRIPTION:Mahatma Gandhi's Birthday
END:VEVENT
BEGIN:VEVENT
SUMMARY:Diwali
DTSTART:20251112
DTEND:20251112
DESCRIPTION:Festival of Lights
END:VEVENT
BEGIN:VEVENT
SUMMARY:Christmas
DTSTART:20251225
DTEND:20251225
DESCRIPTION:Christmas Day
END:VEVENT
END:VCALENDAR"""
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Write sample file
    sample_path = os.path.join("data", "sample_festivals.ics")
    try:
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        return sample_path
    except Exception as e:
        print(f"Error creating sample ICS file: {str(e)}")
        return None

# Test function to verify the fix
def test_festival_loading():
    """Test function to verify festival loading works"""
    try:
        festivals = fetch_festivals_from_ics()
        print(f"✅ Successfully loaded {len(festivals)} festivals")
        
        # Print first few festivals for verification
        for i, (name, start, end) in enumerate(festivals[:5]):
            print(f"{i+1}. {name}: {start.date()} to {end.date()}")
            
        return True
    except Exception as e:
        print(f"❌ Error testing festival loading: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the function when run directly
    test_festival_loading()