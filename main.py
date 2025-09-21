# main.py
import streamlit as st
from modules.firebase_auth import FirebaseAuth
import app  # your dashboard code

# Initialize Firebase Auth
@st.cache_resource
def init_firebase():
    return FirebaseAuth()

firebase_auth = init_firebase()

st.set_page_config(page_title="Goa Police Dashboard", layout="wide")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.auth_mode = "signin"  # signin or signup

# If not logged in → show auth forms
if not st.session_state.logged_in:
    # Center the title and add styling
    st.markdown(
        "<div style='text-align: center; margin-bottom: 2rem;'>"
        "<h1 style='color: #1f4e79; font-size: 3rem; margin-bottom: 0.5rem;'>Goa Police Dashboard</h1>"
        "<p style='color: #666; font-size: 1.2rem; margin-bottom: 0;'>Official Portal for Goa Police Officers</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Create more compact login box
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    
    with col2:
        # Compact auth mode selection
        auth_mode = st.radio(
            "",
            ["Sign In", "Sign Up", "Reset Password"],
            horizontal=True,
            key="auth_mode_radio"
        )
        
        if auth_mode == "Sign In":
            st.markdown(
                "<div style='text-align: center; margin: 1rem 0;'>"
                "<h3 style='color: #1f4e79; margin-bottom: 1rem;'>Officer Sign In</h3>"
                "</div>", 
                unsafe_allow_html=True
            )
            
            with st.form("signin_form", border=True):
                email = st.text_input(
                    "Email Address",
                    placeholder="officer@goapolice.gov.in"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password"
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    submit_btn = st.form_submit_button(
                        "Sign In",
                        use_container_width=True,
                        type="primary"
                    )
                with col_btn2:
                    forgot_btn = st.form_submit_button(
                        "Forgot Password?",
                        use_container_width=True
                    )
                
                if submit_btn:
                    if not email or not password:
                        st.error("Please enter both email and password")
                    else:
                        with st.spinner("Authenticating..."):
                            success, message, user_data = firebase_auth.sign_in(email, password)
                            
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.user = user_data
                                
                                # Log activity
                                firebase_auth.log_activity(
                                    user_data['id'],
                                    'login',
                                    {'method': 'email'}
                                )
                                
                                st.success(f"Welcome, {user_data['full_name']} ({user_data['rank']})!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(message)
                
                if forgot_btn:
                    st.session_state.auth_mode = "reset"
                    st.rerun()
        
        elif auth_mode == "Sign Up":
            st.markdown(
                "<div style='text-align: center; margin: 1rem 0;'>"
                "<h3 style='color: #1f4e79; margin-bottom: 1rem;'>New Officer Registration</h3>"
                "</div>", 
                unsafe_allow_html=True
            )
            
            with st.form("signup_form", border=True):
                # Personal Information
                st.markdown("#### Personal Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    full_name = st.text_input(
                        "Full Name *",
                        placeholder="Enter your full name"
                    )
                    email = st.text_input(
                        "Email Address *",
                        placeholder="officer@goapolice.gov.in"
                    )
                    phone = st.text_input(
                        "Phone Number *",
                        placeholder="+91 XXXXXXXXXX"
                    )
                
                with col2:
                    badge_number = st.text_input(
                        "Badge Number *",
                        placeholder="Enter your badge number"
                    )
                    rank = st.selectbox(
                        "Rank *",
                        [
                            "Constable",
                            "Head Constable",
                            "Assistant Sub-Inspector",
                            "Sub-Inspector",
                            "Inspector",
                            "Deputy Superintendent",
                            "Superintendent",
                            "Deputy Inspector General",
                            "Inspector General"
                        ]
                    )
                    jurisdiction = st.selectbox(
                        "Jurisdiction *",
                        [
                            "North Goa",
                            "South Goa",
                            "Panaji",
                            "Margao",
                            "Vasco",
                            "Mapusa",
                            "Ponda",
                            "Bicholim",
                            "Pernem",
                            "Canacona",
                            "Quepem",
                            "Sanguem"
                        ]
                    )
                
                # Password Section
                st.markdown("#### Security")
                col3, col4 = st.columns(2)
                
                with col3:
                    password = st.text_input(
                        "Password *",
                        type="password",
                        placeholder="Min 8 chars, 1 upper, 1 lower, 1 number, 1 special",
                        help="Password must contain at least 8 characters, including uppercase, lowercase, number and special character"
                    )
                
                with col4:
                    confirm_password = st.text_input(
                        "Confirm Password *",
                        type="password",
                        placeholder="Re-enter your password"
                    )
                
                # Terms and Conditions
                st.markdown("#### Terms & Conditions")
                terms = st.checkbox(
                    "I agree to the terms and conditions and understand that my account requires admin verification"
                )
                
                submit_btn = st.form_submit_button(
                    "Register",
                    use_container_width=True,
                    type="primary"
                )
                
                if submit_btn:
                    # Validate all fields
                    if not all([full_name, email, password, confirm_password, badge_number, phone]):
                        st.error("Please fill in all required fields")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    elif not terms:
                        st.error("Please accept the terms and conditions")
                    else:
                        with st.spinner("Creating your account..."):
                            success, message, user_data = firebase_auth.sign_up(
                                email=email,
                                password=password,
                                full_name=full_name,
                                rank=rank,
                                jurisdiction=jurisdiction,
                                badge_number=badge_number,
                                phone=phone
                            )
                            
                            if success:
                                st.success(message)
                                st.info("Please wait for admin verification. You will be notified once your account is approved.")
                                
                                # Log registration activity
                                if user_data:
                                    firebase_auth.log_activity(
                                        user_data['id'],
                                        'registration',
                                        {'method': 'email'}
                                    )
                            else:
                                st.error(message)
        
        elif auth_mode == "Reset Password":
            st.markdown(
                "<div style='text-align: center; margin: 1rem 0;'>"
                "<h3 style='color: #1f4e79; margin-bottom: 1rem;'>Reset Password</h3>"
                "</div>", 
                unsafe_allow_html=True
            )
            
            with st.form("reset_form", border=True):
                email = st.text_input(
                    "Enter your registered email",
                    placeholder="officer@goapolice.gov.in"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    submit_btn = st.form_submit_button(
                        "Send Reset Link",
                        use_container_width=True,
                        type="primary"
                    )
                with col2:
                    back_btn = st.form_submit_button(
                        "Back to Sign In",
                        use_container_width=True
                    )
                
                if submit_btn:
                    if not email:
                        st.error("Please enter your email address")
                    else:
                        with st.spinner("Processing..."):
                            success, message = firebase_auth.reset_password(email)
                            
                            if success:
                                st.success("Password reset instructions sent to your email")
                                st.info(message)  # In production, remove this
                            else:
                                st.error(message)
                
                if back_btn:
                    st.session_state.auth_mode = "signin"
                    st.rerun()
        
        # Footer
        st.markdown(
            "<div style='text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd;'>"
            "<p style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>Secure authentication powered by Firebase</p>"
            "<p style='color: #666; font-size: 0.9rem; margin-bottom: 0;'>For support, contact: support@goapolice.gov.in</p>"
            "</div>", 
            unsafe_allow_html=True
        )

else:
    # Already logged in → show dashboard
    st.sidebar.success(f"Officer: {st.session_state.user['full_name']}")
    st.sidebar.info(f"**Rank:** {st.session_state.user['rank']}")
    st.sidebar.info(f"**Jurisdiction:** {st.session_state.user['jurisdiction']}")
    st.sidebar.info(f"**Badge:** {st.session_state.user['badge_number']}")
    
    st.sidebar.markdown("---")
    
    # Profile Management
    with st.sidebar.expander("Profile Settings"):
        if st.button("Change Password", use_container_width=True):
            st.session_state.show_change_password = True
        
        if st.button("Update Profile", use_container_width=True):
            st.session_state.show_update_profile = True
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        # Log activity before logout
        firebase_auth.log_activity(
            st.session_state.user['id'],
            'logout',
            None
        )
        
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()
    
    # Handle profile modals
    if st.session_state.get('show_change_password', False):
        with st.container():
            st.markdown("### Change Password")
            with st.form("change_password_form"):
                old_password = st.text_input("Current Password", type="password")
                new_password = st.text_input(
                    "New Password",
                    type="password",
                    help="Min 8 chars, 1 upper, 1 lower, 1 number, 1 special"
                )
                confirm_new_password = st.text_input("Confirm New Password", type="password")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Change Password", use_container_width=True):
                        if new_password != confirm_new_password:
                            st.error("New passwords do not match")
                        else:
                            success, message = firebase_auth.change_password(
                                st.session_state.user['id'],
                                old_password,
                                new_password
                            )
                            if success:
                                st.success(message)
                                st.session_state.show_change_password = False
                                st.rerun()
                            else:
                                st.error(message)
                
                with col2:
                    if st.form_submit_button("Cancel", use_container_width=True):
                        st.session_state.show_change_password = False
                        st.rerun()
    
    elif st.session_state.get('show_update_profile', False):
        with st.container():
            st.markdown("### Update Profile")
            with st.form("update_profile_form"):
                user = st.session_state.user
                
                full_name = st.text_input("Full Name", value=user.get('full_name', ''))
                phone = st.text_input("Phone Number", value=user.get('phone', ''))
                jurisdiction = st.selectbox(
                    "Jurisdiction",
                    [
                        "North Goa", "South Goa", "Panaji", "Margao",
                        "Vasco", "Mapusa", "Ponda", "Bicholim",
                        "Pernem", "Canacona", "Quepem", "Sanguem"
                    ],
                    index=0 if user.get('jurisdiction') not in [
                        "North Goa", "South Goa", "Panaji", "Margao",
                        "Vasco", "Mapusa", "Ponda", "Bicholim",
                        "Pernem", "Canacona", "Quepem", "Sanguem"
                    ] else [
                        "North Goa", "South Goa", "Panaji", "Margao",
                        "Vasco", "Mapusa", "Ponda", "Bicholim",
                        "Pernem", "Canacona", "Quepem", "Sanguem"
                    ].index(user.get('jurisdiction'))
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Update", use_container_width=True):
                        update_data = {
                            'full_name': full_name,
                            'phone': phone,
                            'jurisdiction': jurisdiction
                        }
                        
                        success, message = firebase_auth.update_user_profile(
                            st.session_state.user['id'],
                            update_data
                        )
                        
                        if success:
                            # Update session state
                            st.session_state.user.update(update_data)
                            st.success(message)
                            st.session_state.show_update_profile = False
                            st.rerun()
                        else:
                            st.error(message)
                
                with col2:
                    if st.form_submit_button("Cancel", use_container_width=True):
                        st.session_state.show_update_profile = False
                        st.rerun()
    
    else:
        # Run your existing dashboard
        app.run()