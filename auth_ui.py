import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth
import re

# Updated Police ranks for signup
POLICE_RANKS = [
    "Director General of Police",
    "Inspector General of Police",
    "Dy. Inspector General of Police",
    "Deputy Commissioner of Police (Selection Grade)",
    "Deputy Commissioner of Police (Junior Management Grade)",
    "Assistant Superintendent of Police",
    "Dy. Superintendent of Police (SDPO)",
    "Police Inspector (P.I.)",
    "Assistant Police Inspector (A.P.I.)",
    "Police Sub Inspector (P.S.I.)",
    "Assistant Police Sub Inspector (A.S.I.)",
    "Head Constable (H.C.)"
]

TOP_OFFICER_RANKS = POLICE_RANKS[:5]  # First 5 ranks

# Assistant officer ranks that can access caller entry
ASSISTANT_OFFICER_RANKS = [
    "Assistant Police Inspector (A.P.I.)",
    "Police Sub Inspector (P.S.I.)",
    "Assistant Police Sub Inspector (A.S.I.)",
    "Head Constable (H.C.)"
]

def initialize_session_state():
    """Initialize session state variables for authentication"""
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'show_profile_menu' not in st.session_state:
        st.session_state.show_profile_menu = False
    if 'profile_action' not in st.session_state:
        st.session_state.profile_action = None

def validate_username(username):
    """Validate username format"""
    pattern = r'^[a-zA-Z0-9_]{4,20}$'
    return re.match(pattern, username) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    return True, ""

def signup_form():
    """Handle user signup"""
    with st.form("signup_form"):
        st.subheader("Create New Account")
        
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name")
        with col2:
            last_name = st.text_input("Last Name")
            
        username = st.text_input("Username")
        rank = st.selectbox("Select Rank", options=POLICE_RANKS)

        # Police station selection for registration
        station_category = st.selectbox(
            "Select Police Station Category",
            ["North District police stations", "South District police stations", "Other police Stations"]
        )

        if station_category == "North District police stations":
            police_station = st.selectbox("Select Police Station", NORTH_DISTRICT_POLICE_STATIONS)
        elif station_category == "South District police stations":
            police_station = st.selectbox("Select Police Station", SOUTH_DISTRICT_POLICE_STATIONS)
        else:
            police_station = st.selectbox("Select Police Station", OTHER_POLICE_STATIONS)
        
        col3, col4 = st.columns(2)
        with col3:
            password = st.text_input("Create Password", type="password")
        with col4:
            confirm_password = st.text_input("Confirm Password", type="password")
            
        submit = st.form_submit_button("Create Account")

        if submit:
            if not first_name or not last_name:
                st.error("Please enter your full name")
                return
                
            if not validate_username(username):
                st.error("Username must be 4-20 characters long and contain only letters, numbers, and underscores")
                return
            
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            
            is_valid_password, msg = validate_password(password)
            if not is_valid_password:
                st.error(msg)
                return
            
            try:
                # Create user in Firebase Authentication
                user = auth.create_user(
                    uid=username,  # Using username as UID for simplicity
                    password=password
                )
                
                # Create password hash for verification
                import hashlib
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                # Store additional user data in Firestore
                db = firestore.client()
                db.collection('users').document(username).set({
                    'first_name': first_name,
                    'last_name': last_name,
                    'username': username,
                    'password_hash': password_hash,
                    'rank': rank,
                    'police_station_category': station_category,
                    'police_station': police_station,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'is_active': True,
                    'permissions': {
                        'caller_entry': rank in ASSISTANT_OFFICER_RANKS,
                        'notifications': True,
                        'analytics': True
                    }
                })
                
                st.success("Account created successfully! Please log in.")
            except Exception as e:
                st.error(f"Error creating account: {str(e)}")

def login_form():
    """Handle user login"""
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        submit = st.form_submit_button("Login")

        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return
                
            try:
                # Try Firebase Authentication first
                try:
                    import firebase_admin.auth as firebase_auth
                    # Verify user exists in Firebase Auth
                    user_record = firebase_auth.get_user(username)
                    
                    # Create a custom token to verify password
                    custom_token = firebase_auth.create_custom_token(username)
                    
                    # For now, we'll use a simple verification method
                    # In production, you'd use Firebase Client SDK for password verification
                    
                except Exception as auth_error:
                    st.error("Invalid username or password")
                    return
                
                # Get user data from Firestore
                db = firestore.client()
                user_doc = db.collection('users').document(username).get()
                
                if not user_doc.exists:
                    st.error("Invalid username or password")
                    return
                
                user_data = user_doc.to_dict()
                
                # Password verification
                import hashlib
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                stored_hash = user_data.get('password_hash')
                
                # If no stored hash exists, require password reset
                if not stored_hash:
                    st.error("Account needs password reset. Please contact administrator.")
                    return
                
                # Verify password
                if stored_hash != password_hash:
                    st.error("Invalid username or password")
                    return
                
                # Check if user account is active
                if not user_data.get('is_active', True):
                    st.error("Your account has been deactivated. Please contact administrator.")
                    return
                
                st.session_state.authentication_status = True
                st.session_state.username = username
                st.session_state.user_id = username
                st.session_state.user_data = user_data
                
                st.success("Login successful!")
                st.rerun()
            except Exception as e:
                st.error("Invalid username or password")

def show_user_info():
    """Show logged in user info with enhanced profile display"""
    if st.session_state.user_data:
        user_data = st.session_state.user_data

        # Enhanced header with user info
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown(f"## Welcome, {user_data.get('first_name', '')} {user_data.get('last_name', '')}")
        
        with col2:
            st.markdown(f"##### Rank: {user_data.get('rank', 'N/A')}")
            st.markdown(f"##### Station: {user_data.get('police_station', 'N/A')}")
        
        with col3:
            # Quick logout button
            if st.button("Logout", type="secondary", help="Click to logout"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Logged out successfully!")
                st.rerun()

        # Access permissions display
        permissions = user_data.get('permissions', {})
        if permissions:
            with st.expander("Your Access Permissions", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    caller_access = "Allowed" if permissions.get('caller_entry', False) else "Not Allowed"
                    st.markdown(f"**Caller Entry:** {caller_access}")
                
                with col2:
                    notif_access = "Allowed" if permissions.get('notifications', True) else "Not Allowed"
                    st.markdown(f"**Notifications:** {notif_access}")
                
                with col3:
                    analytics_access = "Allowed" if permissions.get('analytics', True) else "Not Allowed"
                    st.markdown(f"**Analytics:** {analytics_access}")

        st.markdown("---")

def notification_center():
    """Notification center for top officers to send alerts and sub officers to view them."""
    db = firestore.client()
    user_data = st.session_state.user_data
    rank = user_data.get("rank", "")

    st.markdown("<h1 style='font-size: 28px;'>Notification Center</h1>", unsafe_allow_html=True)

    # Top officers can create alerts
    if rank in TOP_OFFICER_RANKS:
        st.markdown("<h3 style='font-size: 22px;'>Create Alert for Sub Officers</h3>", unsafe_allow_html=True)
        
        with st.expander("Create New Alert", expanded=True):
            with st.form("create_alert_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    alert_title = st.text_input("Alert Title *", placeholder="Emergency Alert")
                    caller_name = st.text_input("Caller Name", placeholder="Name of the person calling")
                    location = st.text_input("Deployment Location", placeholder="Location for deployment (optional)")
                
                with col2:
                    alert_priority = st.selectbox("Priority Level", ["High", "Medium", "Low"])
                    caller_location = st.text_input("Caller Location", placeholder="Location of the caller")
                    alert_type = st.selectbox("Alert Type", ["Emergency", "Information", "Warning", "Update"])
                
                alert_message = st.text_area("Alert Message *", placeholder="Detailed alert information", height=100)
                
                col_submit, col_clear = st.columns(2)
                
                with col_submit:
                    submit_alert = st.form_submit_button("Send Alert", type="primary", use_container_width=True)
                
                with col_clear:
                    clear_form = st.form_submit_button("Clear Form", use_container_width=True)

                if submit_alert and alert_title and alert_message:
                    try:
                        # Store alert in Firestore
                        alert_data = {
                            "title": alert_title,
                            "message": alert_message,
                            "caller_name": caller_name or "N/A",
                            "caller_location": caller_location or "N/A",
                            "location": location or "N/A",
                            "priority": alert_priority,
                            "alert_type": alert_type,
                            "from_officer": user_data["username"],
                            "from_rank": rank,
                            "from_name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}",
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "active": True,
                            "created_date": firestore.SERVER_TIMESTAMP
                        }
                        
                        db.collection("alerts").add(alert_data)
                        st.success("Alert sent successfully to all sub officers!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error sending alert: {str(e)}")

        # Show recent alerts sent by this officer
        st.markdown("<h3 style='font-size: 20px;'>My Recent Alerts</h3>", unsafe_allow_html=True)
        try:
            recent_alerts = db.collection("alerts").where("from_officer", "==", user_data["username"]).limit(5).stream()
            alert_count = 0
            
            for alert in recent_alerts:
                alert_data = alert.to_dict()
                alert_count += 1
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"<h5 style='margin-bottom: 5px; font-weight: bold;'>{alert_data.get('title', 'N/A')}</h5>", unsafe_allow_html=True)
                        st.caption(f"{alert_data.get('message', 'N/A')[:100]}...")
                    
                    with col2:
                        st.markdown(f"**Priority:** {alert_data.get('priority', 'Low')}")
                    
                    with col3:
                        if st.button("Deactivate", key=f"deactivate_{alert.id}", help="Deactivate alert"):
                            db.collection("alerts").document(alert.id).update({"active": False})
                            st.rerun()
                    
                    st.markdown("---")
            
            if alert_count == 0:
                st.info("No recent alerts found.")
                
        except Exception as e:
            st.error(f"Error loading recent alerts: {str(e)}")

    # Sub officers see alerts
    else:
        st.markdown("<h3 style='font-size: 20px;'>Active Alerts from Command</h3>", unsafe_allow_html=True)
        try:
            # Simple query without complex ordering
            alerts_ref = db.collection("alerts").where("active", "==", True).limit(15)
            alerts = alerts_ref.stream()
            
            # Process results
            alert_list = []
            for alert in alerts:
                alert_data = alert.to_dict()
                alert_data['id'] = alert.id
                if 'timestamp' in alert_data:
                    alert_list.append(alert_data)
            
            # Sort by timestamp if available
            if alert_list:
                try:
                    alert_list.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                except:
                    pass
            
            # Display alerts with enhanced UI
            if alert_list:
                for i, alert_data in enumerate(alert_list[:10]):  # Show latest 10
                    priority = alert_data.get('priority', 'Medium')
                    alert_type = alert_data.get('alert_type', 'Information')
                    
                    # Color coding based on priority
                    if priority == 'High':
                        border_color = "#ff4444"
                    elif priority == 'Medium':
                        border_color = "#ffaa00"
                    else:
                        border_color = "#44ff44"
                    
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {border_color}; padding-left: 10px; margin-bottom: 15px;">
                            """,
                            unsafe_allow_html=True
                        )
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"<h4 style='margin-bottom: 5px; font-weight: bold;'>{alert_data.get('title', 'N/A')}</h4>", unsafe_allow_html=True)
                            st.markdown(f"**Message:** {alert_data.get('message', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**Priority:** {priority}")
                            st.markdown(f"**Type:** {alert_type}")
                        
                        with col3:
                            st.markdown(f"**From:** {alert_data.get('from_name', 'N/A')}")
                            st.caption(f"({alert_data.get('from_rank', 'N/A')})")
                        
                        # Additional details in expandable section
                        with st.expander("More Details"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown(f"**Caller Name:** {alert_data.get('caller_name', 'N/A')}")
                                st.markdown(f"**Caller Location:** {alert_data.get('caller_location', 'N/A')}")
                            
                            with col_b:
                                st.markdown(f"**Deployment Location:** {alert_data.get('location', 'N/A')}")
                                st.markdown(f"**Alert Type:** {alert_data.get('alert_type', 'N/A')}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")
            else:
                st.info("No active alerts at the moment.")
                
        except Exception as e:
            st.error(f"Error loading alerts: {str(e)}")
            st.info("Please try refreshing the page.")

# Police station lists
NORTH_DISTRICT_POLICE_STATIONS = [
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
    "Mandrem Police Station"
]

SOUTH_DISTRICT_POLICE_STATIONS = [
    "Margao Police Station",
    "Colva Police Station",
    "Curchorem Police Station",
    "Canacona Police Station",
    "Sanguem Police Station",
    "Quepem Police Station",
    "Maina Curtorim Police Station",
    "Fatorda Police Station",
    "Ponda Police Station"
]

OTHER_POLICE_STATIONS = [
    "Traffic Police Station",
    "Cyber Crime Police Station",
    "Women Police Station"
]