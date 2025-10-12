# caller_entry.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timezone, date, time, timedelta
import uuid
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from goa_locations import GOA_LOCATIONS, GOA_SUBSTATIONS

# Assistant police officer ranks that can access caller entry
ASSISTANT_OFFICER_RANKS = [
    "Assistant Police Inspector (A.P.I.)",
    "Police Sub Inspector (P.S.I.)",
    "Assistant Police Sub Inspector (A.S.I.)",
    "Head Constable (H.C.)"
]

# Signal types for the dropdown
SIGNAL_TYPES = [
    "Created Event",
    "External Source", 
    "Missed call",
    "SMS",
    "Voice call"
]

# Event main types
EVENT_MAIN_TYPES = [
    "Crime",
    "Accident",
    "Medical Emergency",
    "Fire",
    "Traffic",
    "Public Order",
    "Domestic Dispute",
    "Theft/Robbery",
    "Missing Person",
    "Noise Complaint",
    "Women Harassment",
    "Child Harassment",
    "Drink and Drove",
    "Drugs Associated",
    "Fraud or cheating",
    "Fighting and assault",
    "Kidnapping",
    "Murder",
    "Property Offense",
    "Robbery",
    "Suicide",
    "Terrorist attack",
    "Threat",
    "Other"
]

# Station main options (based on your existing police stations)
STATION_MAIN_OPTIONS = [
    "Panaji Police Station",
    "Old Goa Police Station",
    "Agacaim Police Station",
    "Mapusa Police Station",
    "Anjuna Police Station",
    "Pernem Police Station",
    "Colvale Police Station",
    "Porvorim Police Station",
    "Calangute Police Station",
    "Saligao Police Station",
    "Bicholim Police Station",
    "Valpoi Police Station",
    "Mopa Police Station",
    "Mandrem Police Station",
    "Margao Police Station",
    "Colva Police Station",
    "Curchorem Police Station",
    "Canacona Police Station",
    "Sanguem Police Station",
    "Quepem Police Station",
    "Maina Curtorim Police Station",
    "Fatorda Police Station",
    "Ponda Police Station",
    "Traffic Police Station",
    "Cyber Crime Police Station",
    "Women Police Station"
]

def check_access_permission(user_data):
    """Check if the user has permission to access caller entry"""
    if not user_data:
        return False
    
    user_rank = user_data.get('rank', '')
    return user_rank in ASSISTANT_OFFICER_RANKS

def generate_event_id():
    """Generate a unique event ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8].upper()
    return f"EVT-{timestamp}-{unique_id}"

def geocode_location(location_name):
    """Get coordinates for a location using geocoding"""
    try:
        geolocator = Nominatim(user_agent="goa-police-app/1.0 (raghavshrivastav0812@gmail.com)", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
        
        query = f"{location_name}, Goa, India"
        location = geocode(query)
        
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.warning(f"Could not geocode location: {e}")
        return None, None

def save_caller_entry(entry_data, officer_info):
    """Save caller entry to Firestore"""
    try:
        db = firestore.client()
        
        # Add metadata
        entry_data.update({
            'created_by': officer_info.get('username', ''),
            'created_by_rank': officer_info.get('rank', ''),
            'created_by_station': officer_info.get('police_station', ''),
            'created_at': firestore.SERVER_TIMESTAMP,
            'status': 'active'
        })
        
        # Save to Firestore
        doc_ref = db.collection('caller_entries').add(entry_data)
        return True, f"Entry saved successfully with ID: {doc_ref[1].id}"
    
    except Exception as e:
        return False, f"Error saving entry: {str(e)}"

def get_caller_entries(officer_username=None, limit=50):
    """Retrieve caller entries from Firestore"""
    try:
        db = firestore.client()
        
        if officer_username:
            # Get entries created by specific officer
            entries_ref = db.collection('caller_entries').where('created_by', '==', officer_username).limit(limit)
        else:
            # Get all entries
            entries_ref = db.collection('caller_entries').limit(limit)
        
        entries = entries_ref.stream()
        
        entries_list = []
        for entry in entries:
            entry_data = entry.to_dict()
            entry_data['doc_id'] = entry.id
            entries_list.append(entry_data)
        
        return entries_list
    
    except Exception as e:
        st.error(f"Error retrieving entries: {str(e)}")
        return []

def caller_entry_form():
    """Display the caller entry form"""
    st.header("Caller Entry Form")
    st.markdown("Enter details of the incoming call/complaint")
    
    # Initialize session state for time tracking
    if 'mdt_assigned_time' not in st.session_state:
        st.session_state.mdt_assigned_time = None
    if 'delivered_seconds' not in st.session_state:
        st.session_state.delivered_seconds = 0
    if 'reach_seconds' not in st.session_state:
        st.session_state.reach_seconds = 0
        
    with st.form("caller_entry_form", clear_on_submit=False):
        # Row 1: Basic Information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_date = st.date_input(
                "Create Date *",
                help="Date when the call was received"
            )
            create_time = st.time_input(
                "Create Time *",
                help="Time when the call was received"
            )
        
        with col2:
            signal_type = st.selectbox(
                "Signal Type *",
                options=SIGNAL_TYPES,
                help="Type of signal/call received"
            )
        
        with col3:
            event_id = st.text_input(
                "Event ID",
                value=generate_event_id(),
                help="Auto-generated Event ID"
            )
        
        # Row 2: Caller Information
        col4, col5 = st.columns(2)
        
        with col4:
            dialled_no = st.text_input(
                "Dialled Number *",
                placeholder="112",
                help="Number dialled by caller (usually 112)"
            )
        
        with col5:
            caller_name = st.text_input(
                "Caller Name",
                placeholder="Enter caller's full name",
                help="Full name of the person calling (optional)"
            )
        
        # Row 3: Event Details
        col6, col7 = st.columns(2)
        
        with col6:
            event_main_type = st.selectbox(
                "Event Main Type *",
                options=EVENT_MAIN_TYPES,
                help="Primary category of the incident"
            )
        
        with col7:
            incident_location = st.selectbox(
                "Incident Location *",
                options=[''] + sorted(GOA_LOCATIONS),
                format_func=lambda x: 'Select incident location...' if x == '' else x,
                help="Select the location where incident occurred"
            )
        
        # Row 4: Event Information (Full Width)
        event_information = st.text_area(
            "Event Information *",
            placeholder="Detailed description of the incident/complaint",
            height=100,
            help="Comprehensive details about the incident"
        )
        
        # Row 5: Station Information
        col8, col9, col10 = st.columns(3)
        
        with col8:
            station_main = st.selectbox(
                "Main Station *",
                options=STATION_MAIN_OPTIONS,
                help="Primary police station handling the case"
            )
        
        with col9:
            station_sub = st.selectbox(
                "Sub Station",
                options=[''] + sorted(GOA_SUBSTATIONS),
                format_func=lambda x: 'Select sub-station...' if x == '' else x,
                help="Sub-station or outpost (if applicable)"
            )
        
        with col10:
            call_sign = st.text_input(
                "Call Sign",
                placeholder="Unit call sign",
                help="Radio call sign of responding unit"
            )
        
        # Row 6: Time Tracking with automatic calculation
        st.subheader("Response Time Tracking")
        col11, col12, col13 = st.columns(3)
        
        with col11:
            col_time, col_button = st.columns([3, 1])
            with col_time:
                # Use session state for the time input
                mdt_assigned_time = st.time_input(
                    "MDT Assigned Time",
                    value=st.session_state.mdt_assigned_time,
                    help="Time when Mobile Data Terminal was assigned"
                )
            with col_button:
                set_time = st.form_submit_button(
                    "Allot Time",
                    help="Set MDT Assigned Time to current time"
                )
                if set_time:
                    st.session_state.mdt_assigned_time = datetime.now().time()
        
        with col12:
            col_delivered_time, col_delivered_btn = st.columns([3, 1])
            with col_delivered_time:
                mins, secs = divmod(st.session_state.delivered_seconds, 60)
                st.text_input(
                    "Delivered Time",
                    value=f"{mins} min {secs} sec",
                    disabled=True,
                    help="Time since MDT Assigned Time"
                )
                
            with col_delivered_btn:
                delivered_btn = st.form_submit_button("Delivered", help="Set to current time difference from MDT Assigned Time")
                if delivered_btn:
                    if st.session_state.mdt_assigned_time is not None:
                        current_time = datetime.now().time()
                        base_date = date.today()
                        mdt_dt = datetime.combine(base_date, st.session_state.mdt_assigned_time)
                        current_dt = datetime.combine(base_date, current_time)
                        
                        if current_time < st.session_state.mdt_assigned_time:
                            current_dt = current_dt + timedelta(days=1)
                        
                        st.session_state.delivered_seconds = int((current_dt - mdt_dt).total_seconds())
        
        with col13:
            col_reach_time, col_reach_btn = st.columns([3, 1])
            with col_reach_time:
                mins, secs = divmod(st.session_state.reach_seconds, 60)
                st.text_input(
                    "Reach Time",
                    value=f"{mins} min {secs} sec",
                    disabled=True,
                    help="Time since MDT Assigned Time"
                )
                
            with col_reach_btn:
                reach_btn = st.form_submit_button("Reach", help="Set to current time difference from MDT Assigned Time")
                if reach_btn:
                    if st.session_state.mdt_assigned_time is not None:
                        current_time = datetime.now().time()
                        base_date = date.today()
                        mdt_dt = datetime.combine(base_date, st.session_state.mdt_assigned_time)
                        current_dt = datetime.combine(base_date, current_time)
                        
                        if current_time < st.session_state.mdt_assigned_time:
                            current_dt = current_dt + timedelta(days=1)
                        
                        st.session_state.reach_seconds = int((current_dt - mdt_dt).total_seconds())
        
        # Calculate MDT Response Time automatically
        col14 = st.columns(1)[0]
        with col14:
            response_seconds = max(0, st.session_state.reach_seconds - st.session_state.delivered_seconds)
            mins, secs = divmod(response_seconds, 60)
            
            st.text_input(
                "MDT Response Time - Auto Calculated",
                value=f"{mins} min {secs} sec",
                disabled=True,
                help="Response time (Reach time - Delivered time)"
            )
        
        # Row 7: Action and Closure
        col15 = st.columns(1)[0]
        with col15:
            action_taken_at_dcc = st.text_input(
                "Action Taken at DCC",
                placeholder="Action taken at District Control Center",
                help="Actions taken by the control center"
            )
        
        # Row 8: Closure Comments (Full Width)
        closure_comments = st.text_area(
            "Closure Comments",
            placeholder="Final remarks and closure details",
            height=80,
            help="Comments when closing the case"
        )
        
        # Form submission buttons
        col_submit, col_clear = st.columns([2, 1])
        
        with col_submit:
            submitted = st.form_submit_button(
                "Save Entry",
                use_container_width=True,
                type="primary"
            )
        
        with col_clear:
            clear_form = st.form_submit_button(
                "Clear Form",
                use_container_width=True
            )
            
            if clear_form:
                # Clear the session state for the form
                if 'mdt_assigned_time' in st.session_state:
                    del st.session_state.mdt_assigned_time
                if 'delivered_seconds' in st.session_state:
                    del st.session_state.delivered_seconds
                if 'reach_seconds' in st.session_state:
                    del st.session_state.reach_seconds
                st.rerun()
        
        if submitted:
            # Validate required fields (caller_name is no longer required)
            required_fields = {
                'Create Date': create_date,
                'Create Time': create_time,
                'Signal Type': signal_type,
                'Dialled Number': dialled_no,
                'Event Main Type': event_main_type,
                'Event Information': event_information,
                'Incident Location': incident_location,
                'Main Station': station_main
            }
            
            missing_fields = [field for field, value in required_fields.items() if not value]
            
            if missing_fields:
                st.error(f"Please fill in the following required fields: {', '.join(missing_fields)}")
            else:
                # Combine date and time for create_time
                create_datetime = datetime.combine(create_date, create_time)
                
                # Calculate response time
                response_seconds = max(0, st.session_state.reach_seconds - st.session_state.delivered_seconds)
                
                # Geocode the incident location to get coordinates
                with st.spinner("Getting location coordinates..."):
                    caller_lat, caller_lon = geocode_location(incident_location)
                
                # Prepare data for saving
                entry_data = {
                    'sl_no': len(get_caller_entries()) + 1,  # Simple serial number
                    'create_time': create_datetime.isoformat(),
                    'signal_type': signal_type,
                    'event_id': event_id,
                    'dialled_no': dialled_no,
                    'caller_name': caller_name if caller_name else "Not Provided",
                    'event_main_type': event_main_type,
                    'event_information': event_information,
                    'incident_location': incident_location,
                    'caller_lat': caller_lat,
                    'caller_lon': caller_lon,
                    'station_main': station_main,
                    'station_sub': station_sub,
                    'call_sign': call_sign,
                    'mdt_assigned_time': mdt_assigned_time.isoformat() if mdt_assigned_time else None,
                    'delivered_seconds': st.session_state.delivered_seconds,
                    'reach_seconds': st.session_state.reach_seconds,
                    'mdt_response_time_seconds': response_seconds,
                    'action_taken_at_dcc': action_taken_at_dcc,
                    'closure_comments': closure_comments
                }
                
                # Save the entry
                success, message = save_caller_entry(entry_data, st.session_state.user_data)
                
                if success:
                    if caller_lat and caller_lon:
                        st.success(f"{message}\n📍 Location coordinates: {caller_lat:.4f}, {caller_lon:.4f}")
                    else:
                        st.success(f"{message}\n⚠️ Location coordinates could not be determined")
                    st.balloons()
                    # Optionally rerun to clear the form
                    st.rerun()
                else:
                    st.error(message)

def view_caller_entries():
    """Display saved caller entries"""
    st.header("Caller Entries Log")
    
    # Options for viewing
    view_option = st.radio(
        "View entries:",
        ["My Entries", "All Entries"],
        horizontal=True
    )
    
    # Get entries based on selection
    if view_option == "My Entries":
        entries = get_caller_entries(officer_username=st.session_state.user_data.get('username'))
    else:
        entries = get_caller_entries()
    
    if entries:
        # Convert to DataFrame for better display
        df = pd.DataFrame(entries)
        
        # Display summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entries", len(entries))
        
        with col2:
            crime_count = len([e for e in entries if e.get('event_main_type') == 'Crime'])
            st.metric("Crime Cases", crime_count)
        
        with col3:
            emergency_count = len([e for e in entries if e.get('signal_type') == 'Emergency'])
            st.metric("Emergency Calls", emergency_count)
        
        with col4:
            try:
                # Handle both old and new datetime formats
                today_entries = []
                today_date = datetime.now().date()
                
                for entry in entries:
                    create_time_str = entry.get('create_time', '')
                    if create_time_str:
                        try:
                            # Try parsing as full datetime
                            entry_datetime = pd.to_datetime(create_time_str)
                            if entry_datetime.date() == today_date:
                                today_entries.append(entry)
                        except:
                            # If parsing fails, skip this entry for today's count
                            continue
                
                today_count = len(today_entries)
                st.metric("Today's Entries", today_count)
            except:
                st.metric("Today's Entries", 0)
        
        st.markdown("---")
        
        # Search and filter options
        search_term = st.text_input("Search entries (Event ID, Caller Name, Location)")
        
        if search_term:
            filtered_entries = [
                entry for entry in entries
                if search_term.lower() in str(entry.get('event_id', '')).lower() or
                   search_term.lower() in str(entry.get('caller_name', '')).lower() or
                   search_term.lower() in str(entry.get('incident_location', '')).lower()
            ]
        else:
            filtered_entries = entries
        
        # Display entries in expandable cards
        for i, entry in enumerate(reversed(filtered_entries[-20:])):  # Show latest 20
            with st.expander(
                f"{entry.get('event_id', 'N/A')} - {entry.get('caller_name', 'N/A')} - {entry.get('event_main_type', 'N/A')}",
                expanded=False
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Create Time:** {entry.get('create_time', 'N/A')}")
                    st.markdown(f"**Signal Type:** {entry.get('signal_type', 'N/A')}")
                    st.markdown(f"**Caller Name:** {entry.get('caller_name', 'N/A')}")
                    st.markdown(f"**Dialled Number:** {entry.get('dialled_no', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Event Type:** {entry.get('event_main_type', 'N/A')}")
                    st.markdown(f"**Location:** {entry.get('incident_location', 'N/A')}")
                    st.markdown(f"**Main Station:** {entry.get('station_main', 'N/A')}")
                    st.markdown(f"**Call Sign:** {entry.get('call_sign', 'N/A')}")
                
                with col3:
                    st.markdown(f"**Response Time:** {entry.get('mdt_response_time', 'N/A')} min")
                    st.markdown(f"**Created By:** {entry.get('created_by', 'N/A')}")
                    st.markdown(f"**Rank:** {entry.get('created_by_rank', 'N/A')}")
                
                st.markdown(f"**Event Information:** {entry.get('event_information', 'N/A')}")
                
                if entry.get('action_taken_at_dcc'):
                    st.markdown(f"**DCC Action:** {entry.get('action_taken_at_dcc')}")
                
                if entry.get('closure_comments'):
                    st.markdown(f"**Closure Comments:** {entry.get('closure_comments')}")
        
        # Export option
        st.markdown("---")
        if st.button("Export to CSV", help="Download entries as CSV file"):
            df_export = pd.DataFrame(filtered_entries)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"caller_entries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No caller entries found. Start by creating a new entry.")

def caller_entry_dashboard():
    """Main caller entry dashboard"""
    user_data = st.session_state.user_data
    
    if not check_access_permission(user_data):
        st.error("Access Denied: Only Assistant Police Officers can access this section.")
        st.info("Required ranks: " + ", ".join(ASSISTANT_OFFICER_RANKS))
        return
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["New Entry", "View Entries"])
    
    with tab1:
        caller_entry_form()
    
    with tab2:
        view_caller_entries()