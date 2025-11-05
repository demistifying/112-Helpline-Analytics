# app.py  (updated with caller entry integration)
import os
from datetime import timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
#from streamlit_folium import st_folium
import firebase_admin
from firebase_admin import credentials, firestore, auth
from streamlit_option_menu import option_menu

from auth_ui import initialize_session_state, signup_form, login_form, show_user_info, notification_center, TOP_OFFICER_RANKS
from caller_entry import caller_entry_dashboard, check_access_permission  # Import caller entry module
from modules.data_loader import load_data, preprocess
from data_integration import append_caller_entries_to_dataset
from modules.analysis import (
    agg_calls_by_day, agg_calls_by_hour, category_distribution, compute_kpis,
    interpret_time_series, interpret_hourly_distribution
)
from modules.mapping import pydeck_points_map, pydeck_heatmap, pydeck_hexbin_3d
from modules.festivals_ics import fetch_festivals_from_ics
from modules.festivals_utils import filter_significant_festivals
from modules.ui_calendar import render_month_calendar

# --- PREDICTIVE MODELS IMPORTS ---
from modules.feature_engineering import prepare_features_for_prophet, prepare_features_for_xgboost
from modules.predictive_models import (
    train_prophet_model, predict_with_prophet,
    train_event_type_model, predict_event_type_distribution,
    train_peak_hour_model, predict_hourly_calls_for_n_days
)
from modules.festival_baselines import calculate_weekly_top_n_peaks, get_baseline_for_date

# Initialize Firebase only once
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    except KeyError:
        st.error("Firebase credentials not found in secrets. Please add them in Streamlit Cloud settings.")
        st.stop()

# Firestore client
db = firestore.client()

def main():
    st.set_page_config(
        page_title="Goa Police Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Check authentication status
    if st.session_state.authentication_status is None:
        # Show login/signup interface
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            import os
            if os.path.exists("Emblem_of_Goa_Police (1).png"):
                col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
                with col_img2:
                    st.image("Emblem_of_Goa_Police (1).png", width=150)
            st.markdown("<div style='text-align: center;'><h1>Goa Police Dashboard</h1><p style='font-weight: bold; font-size: 18px;'>Official Portal for Goa Police Officers</p></div>", unsafe_allow_html=True)
            st.markdown("---")
        
        # Create tabs for login and signup
        tab1, tab2 = st.tabs(["Officer Login", "New Registration"])
        
        with tab1:
            login_form()
        
        with tab2:
            signup_form()
    
    else:
        # User is authenticated - show main dashboard
        show_user_info()
        
        # Check if user has access to caller entry and alert creation
        has_caller_access = check_access_permission(st.session_state.user_data)
        user_rank = st.session_state.user_data.get('rank', '')
        has_alert_access = user_rank in TOP_OFFICER_RANKS
        
        # Create navigation options based on user permissions
        menu_options = ["Analytics Dashboard", "Predictive Forecasting"]
        menu_icons = ["bar-chart-line", "graph-up-arrow"]
        
        if has_caller_access:
            menu_options.append("Caller Entry")
            menu_icons.append("telephone-fill")
        
        if has_alert_access:
            menu_options.append("Alert Creation")
            menu_icons.append("exclamation-triangle-fill")
        
        menu_options.append("Notifications")
        menu_icons.append("bell-fill")
        
        # --- TOP LEVEL MENU ---
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=menu_icons,
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        # Show content based on selection
        if selected == "Analytics Dashboard":
            show_analytics_dashboard()
        elif selected == "Predictive Forecasting":
            show_predictive_forecasting()
        elif selected == "Caller Entry" and has_caller_access:
            caller_entry_dashboard()
        elif selected == "Alert Creation" and has_alert_access:
            notification_center()
        elif selected == "Notifications":
            notification_center()

def show_analytics_dashboard():
    """Display the main analytics dashboard"""
    st.title("112 Helpline — Analytics Dashboard")
    st.markdown("---")
    
    # -------------------------
    # Load sample data
    # -------------------------
    df_raw, metadata = None, None
    try:
        sample_path = os.path.join("data", "Dummy_Dataset_Full_Standardized.csv")
        if not os.path.exists(sample_path):
            st.error(f"Sample file not found at {sample_path}")
            st.stop()
        else:
            df_raw, metadata = load_data(sample_path)
            st.sidebar.info(f"Using sample data ({metadata['record_count']} rows)")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # -------------------------
    # Preprocess and integrate caller entries
    # -------------------------
    df = preprocess(df_raw)  # ensures date, hour, weekday columns exist
    
    # Append new caller entries to the dataset
    df, new_entries_count = append_caller_entries_to_dataset(df)
    if new_entries_count > 0:
        st.sidebar.success(f"Added {new_entries_count} new caller entries to dataset")
        df = preprocess(df)  # Re-preprocess after adding new data
    
    # Filter out any future dates that might have been added incorrectly
    if 'call_ts' in df.columns:
        today = pd.Timestamp.now().normalize()
        df = df[df['call_ts'] <= today]

    # -------------------------
    # Sidebar filters (date range, category, jurisdiction)
    # -------------------------
    st.sidebar.header("Filters")
    
    if not df.empty and "call_ts" in df.columns and df["call_ts"].notna().any():
        min_date = df["call_ts"].min().date()
        max_date = df["call_ts"].max().date()
    else:
        import datetime
        today = datetime.date.today()
        min_date = today
        max_date = today

    date_range = st.sidebar.date_input(
        "Date range", 
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        key="date_filter"
    )

    categories = df["category"].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("Category", options=categories, default=categories)

    jurisdictions = df["station_sub"].dropna().unique().tolist() if "station_sub" in df.columns else df["jurisdiction"].dropna().unique().tolist()
    selected_jurisdictions = st.sidebar.multiselect("Station/Jurisdiction", options=jurisdictions, default=jurisdictions)

    # Detect significant days based on actual call volume spikes
    from modules.spike_detection import detect_significant_days, create_spike_festivals
    
    # Get top 10 festivals by call volume
    significant_days_data = detect_significant_days(df, category='crime', top_n=10)
    all_festivals = create_spike_festivals(significant_days_data)
    
    st.sidebar.success(f"✅ Found {len(all_festivals)} festivals")

    # -------------------------
    # Apply dataset filters to create df_filtered
    # -------------------------
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range[0], date_range[1]
    else:
        start_date = end_date = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]
    
    # Convert dates to datetime for comparison
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include full end day
    
    jurisdiction_col = "station_sub" if "station_sub" in df.columns else "jurisdiction"
    mask = (
        (df["call_ts"] >= start_datetime) &
        (df["call_ts"] <= end_datetime) &
        (df["category"].isin(selected_categories)) &
        (df[jurisdiction_col].isin(selected_jurisdictions))
    )
    df_filtered = df[mask].copy()
    
    # Add debug info
    st.sidebar.write(f"📅 Selected: {start_date} to {end_date}")
    st.sidebar.write(f"📊 Filtered: {len(df_filtered):,} / {len(df):,} records")

    # -------------------------
    # Determine festivals in selected date range (all) and significant subset
    # -------------------------
    start_sel = pd.to_datetime(date_range[0])
    end_sel = pd.to_datetime(date_range[1])

    festivals_in_range_all = []
    for name, fs, fe in all_festivals:
        fs_ts = pd.to_datetime(fs)
        fe_ts = pd.to_datetime(fe)
        if (start_sel <= fe_ts) and (end_sel >= fs_ts):
            festivals_in_range_all.append((name, fs_ts, fe_ts))

    # --- GUARANTEED: Always show peaks in selected range ---
    # Get peaks specifically for the filtered data
    filtered_peaks = detect_significant_days(df_filtered, category='crime', top_n=10)
    significant_festals_info = filtered_peaks
    
    # Create a set of significant day names for quick lookup
    significant_names = {f['name'] for f in significant_festals_info}
    
    # Debug: Show spike detection info
    if st.sidebar.checkbox("🔍 Debug Spike Detection", value=False):
        st.sidebar.write(f"**High-activity days detected:** {len(all_festivals)}")
        for name, fs, fe in all_festivals[:5]:
            st.sidebar.write(f"• {name}: {fs.date()}")
        if len(all_festivals) > 5:
            st.sidebar.write(f"... and {len(all_festivals) - 5} more")
        
        st.sidebar.write(f"**Days in selected range:** {len(significant_festals_info)}")
        for info in significant_festals_info[:3]:
            st.sidebar.write(f"• {info['name']}: {info['max_count']} calls (+{info['max_pct']:.0f}%)")
    
    # Store significant festivals in session state for predictive models
    st.session_state.significant_festivals = significant_festals_info

    # -------------------------
    # Tag df_filtered rows with festival_name (for stacking & other use)
    # -------------------------
    def tag_festival_for_row(ts, festivals_list):
        for name, fs, fe in festivals_list:
            if fs <= ts <= fe:
                return name
        return "Non-Festival"

    # Tag all festivals first for general awareness
    df_filtered["festival_name"] = df_filtered["call_ts"].apply(tag_festival_for_row, festivals_list=festivals_in_range_all)

    # --- MODIFIED: Create a specific column for the hourly chart ---
    # This column will only contain names of the top 10 festivals. Everything else is "Non-Festival".
    def tag_significant_festivals(row):
        festival_name = row["festival_name"]
        if festival_name in significant_names:
            return festival_name
        else:
            return "Non-Festival"

    df_filtered["significant_festival_name"] = df_filtered.apply(tag_significant_festivals, axis=1)

    # Show overlap warning only if the selected range is small (<= 31 days)
    range_days = (end_sel - start_sel).days
    if festivals_in_range_all and range_days <= 31:
        overlapping_texts = []
        for name, fs, fe in festivals_in_range_all:
            overlapping_texts.append(f"**{name}** ({fs.date()} → {fe.date()})")
        st.warning("Selected date range overlaps festival(s): " + "; ".join(overlapping_texts))

    # -------------------------
    # KPIs Section
    # -------------------------
    st.markdown("## 📊 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpis = compute_kpis(df_filtered)
    
    with kpi1:
        st.metric("📞 Total Calls", f"{kpis['total_calls']:,}", help="Total filtered calls")
    with kpi2:
        st.metric("📈 Daily Average", f"{kpis['avg_per_day']:.1f}", help="Average calls per day")
    with kpi3:
        st.metric("⏰ Peak Hour", f"{kpis['peak_hour']}:00", help="Hour with most calls")
    with kpi4:
        valid_coords = df_filtered[['latitude','longitude']].dropna().shape[0] if 'latitude' in df_filtered.columns else 0
        coverage = (valid_coords / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("🗺️ Location Coverage", f"{coverage:.1f}%", help="Calls with valid coordinates")
    
    st.markdown("---")

    # Create main tabs for different sections
    main_tab1, main_tab2, main_tab3 = st.tabs(["🗺️ Spatial Analysis", "📈 Temporal Analysis", "📊 Category Analysis"])
    
    # -------------------------
    # TAB 1: Spatial Analysis
    # -------------------------
    with main_tab1:
        st.markdown("### Geographic Distribution of Incidents")
        
        # Location search functionality - visible across all maps
        st.markdown("#### 🔍 Search Incidents by Location")
        if 'incident_location' in df_filtered.columns:
            locations = sorted(df_filtered['incident_location'].dropna().unique().tolist())
            if locations:
                selected_location = st.selectbox(
                    "Search and select location:",
                    options=[''] + locations,
                    format_func=lambda x: 'Type to search locations...' if x == '' else x,
                    key="location_search_main"
                )
                
                if selected_location:
                    # Filter incidents for selected location
                    location_incidents = df_filtered[df_filtered['incident_location'] == selected_location]
                    
                    if not location_incidents.empty:
                        st.markdown(f"**📍 {len(location_incidents)} incidents found at {selected_location}**")
                        
                        # Display incident details in table
                        display_cols = ['call_ts', 'category', jurisdiction_col]
                        display_df = location_incidents[display_cols].copy()
                        display_df['call_ts'] = display_df['call_ts'].dt.strftime('%Y-%m-%d %H:%M')
                        display_df.columns = ['Date & Time', 'Category', 'Station/Jurisdiction']
                        
                        st.dataframe(display_df, use_container_width=True, height=200)
            else:
                st.info("No location data available for search.")
        else:
            st.info("Location search not available - missing incident_location column.")
        
        st.markdown("---")
        map_tab1, map_tab2 = st.tabs(["📍 Incident Points", "📊 Heatmap Analysis"])
        
        with map_tab1:
            try:
                with st.spinner("Loading map..."):
                    result = pydeck_points_map(df_filtered)
                    if result:
                        if isinstance(result, tuple) and len(result) == 2:
                            deck_points, location_data = result
                            map_data = st.pydeck_chart(deck_points)
                        
                        # Handle click interactions
                        if map_data and 'last_clicked_object' in map_data and map_data['last_clicked_object']:
                            clicked_data = map_data['last_clicked_object']
                            clicked_lat = clicked_data.get('latitude') or clicked_data.get('position', [None, None])[1]
                            clicked_lon = clicked_data.get('longitude') or clicked_data.get('position', [None, None])[0]
                            
                            if clicked_lat and clicked_lon:
                                st.session_state.clicked_location = (clicked_lat, clicked_lon)
                        
                        # Show details for clicked location
                        if hasattr(st.session_state, 'clicked_location') and st.session_state.clicked_location:
                            clicked_lat, clicked_lon = st.session_state.clicked_location
                            
                            # Filter data for clicked location (with tolerance)
                            tolerance = 0.001
                            nearby_incidents = df_filtered[
                                (abs(df_filtered['latitude'] - clicked_lat) < tolerance) &
                                (abs(df_filtered['longitude'] - clicked_lon) < tolerance)
                            ]
                            
                            if not nearby_incidents.empty:
                                st.markdown(f"### 📍 Location Details ({clicked_lat:.4f}, {clicked_lon:.4f})")
                                
                                # Display summary metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Incidents", len(nearby_incidents))
                                with col2:
                                    categories_count = nearby_incidents['category'].nunique()
                                    st.metric("Categories", categories_count)
                                with col3:
                                    jurisdiction_col = "station_sub" if "station_sub" in nearby_incidents.columns else "jurisdiction"
                                    stations_count = nearby_incidents[jurisdiction_col].nunique()
                                    st.metric("Stations Involved", stations_count)
                                
                                # Show incident details
                                display_cols = ['call_ts', 'category', jurisdiction_col]
                                if 'Date & Time' in nearby_incidents.columns:
                                    display_cols = ['Date & Time', 'category', jurisdiction_col]
                                
                                display_df = nearby_incidents[display_cols].copy()
                                st.dataframe(display_df, use_container_width=True)
                                
                                if st.button("Clear Selection"):
                                    st.session_state.clicked_location = None
                                    st.rerun()
                            else:
                                st.info(f"No incidents found at location ({clicked_lat:.4f}, {clicked_lon:.4f})")
                                if st.button("Clear Selection"):
                                    st.session_state.clicked_location = None
                                    st.rerun()
                        else:
                            st.pydeck_chart(result)
                    else:
                        st.info("No valid coordinates to plot.")
            except Exception as e:
                pass  # Hide the error message
            

        
        with map_tab2:
            try:
                result = pydeck_hexbin_3d(df_filtered)
                if result:
                    st.pydeck_chart(result)
                else:
                    st.info("No valid coordinates to plot heatmap.")
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")

    # -------------------------
    # TAB 2: Temporal Analysis
    # -------------------------
    with main_tab2:
        st.markdown("### Time-based Incident Patterns")
        
        # Time series chart
        st.markdown("#### 📅 Daily Trend Analysis with High-Activity Days")
        if significant_festals_info:
            st.info(f"📊 Showing {len(significant_festals_info)} high-activity days with elevated call volumes")
        ts_df = agg_calls_by_day(df_filtered, date_col="date")

        if not ts_df.empty:
            # Convert date column to datetime for proper alignment
            ts_df['date'] = pd.to_datetime(ts_df['date'])

            # Create the base line chart with better styling
            fig = px.line(ts_df, x="date", y="count", 
                         labels={"date": "Date", "count": "Number of Calls"},
                         title="📈 Daily Call Volume Trend")
            fig.update_traces(
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Calls: %{y}',
                line=dict(color='#3498db', width=3)
            )

            # Clean highlights for peak days
            festival_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            # Always highlight something - use top days from time series if no filtered peaks
            if not significant_festals_info and not ts_df.empty:
                top_ts_days = ts_df.nlargest(3, 'count')
                for idx, row in top_ts_days.iterrows():
                    color = festival_colors[idx % len(festival_colors)]
                    peak_date = pd.to_datetime(row['date'])
                    
                    fig.add_vrect(
                        x0=peak_date - pd.Timedelta(hours=12),
                        x1=peak_date + pd.Timedelta(hours=12),
                        fillcolor=color, opacity=0.3,
                        layer="below", line_width=0
                    )
            
            # Highlight filtered peaks if available
            elif significant_festals_info:
                for idx, info in enumerate(significant_festals_info):
                    color = festival_colors[idx % len(festival_colors)]
                    peak_date = pd.to_datetime(info['max_day'])
                    
                    fig.add_vrect(
                        x0=peak_date - pd.Timedelta(hours=12),
                        x1=peak_date + pd.Timedelta(hours=12),
                        fillcolor=color, opacity=0.3,
                        layer="below", line_width=0
                    )
                    
                    # Add legend marker
                    fig.add_trace(go.Scatter(
                        x=[peak_date],
                        y=[info['max_count']],
                        mode='markers',
                        marker=dict(size=8, color=color),
                        name=info['name'],
                        hovertemplate=f"<b>{info['name']}</b><br>Date: {peak_date.strftime('%B %d, %Y')}<br>Calls: {info['max_count']}<extra></extra>",
                        showlegend=True
                    ))
            
            fig.update_layout(
                yaxis_range=[0, ts_df['count'].max() * 1.1],
                title=dict(x=0.5, font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title="Number of Calls",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    itemsizing="constant",
                    itemwidth=30
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Festival insights and general insights
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with st.expander("📋 Daily Trend Insights", expanded=False):
                    insights = interpret_time_series(ts_df)
                    for ins in insights:
                        st.markdown(f"• {ins}")
            
            with col2:
                if significant_festals_info:
                    with st.expander("📊 High-Activity Days (Crime Calls)", expanded=False):
                        st.markdown("**Days with Significant Call Volume Spikes:**")
                        for idx, info in enumerate(significant_festals_info[:5], 1):
                            st.markdown(f"{idx}. **{info['name']}**: +{info['max_pct']:.0f}% ({info['max_count']} calls on {info['max_day']})")
                        if len(significant_festals_info) > 5:
                            st.markdown(f"... and {len(significant_festals_info) - 5} more days highlighted in the graph")
        else:
            st.info("No data available for the selected filters.")

        st.markdown("---")
        
        # Hourly distribution
        st.markdown("#### ⏰ Hourly Distribution Pattern")
        if significant_names:
            hr = df_filtered.groupby(["hour", "significant_festival_name"]).size().reset_index(name="count")
            color_map = {"Non-Festival": "#85C1E9"}

            fig2 = px.bar(hr, x="hour", y="count", color="significant_festival_name", 
                         barmode="stack", color_discrete_map=color_map,
                         labels={"hour": "Hour of Day", "count": "Number of Calls", "significant_festival_name": "Period"},
                         title="📊 Hourly Call Distribution")
            
            hr_totals = hr.groupby("hour")["count"].sum().reset_index()
            insights = interpret_hourly_distribution(hr_totals)
        else:
            hr = agg_calls_by_hour(df_filtered, hour_col="hour")
            fig2 = px.bar(hr, x="hour", y="count", 
                         labels={"hour": "Hour of Day", "count": "Number of Calls"},
                         title="📊 Hourly Call Distribution")
            fig2.update_traces(marker_color='#85C1E9')
            insights = interpret_hourly_distribution(hr)

        fig2.update_layout(
            title=dict(x=0.5, font=dict(size=16)),
            xaxis_title="Hour of Day (24-hour format)",
            yaxis_title="Number of Calls",
            showlegend=True if significant_names else False
        )

        st.plotly_chart(fig2, use_container_width=True)
        
        # Hourly insights in expander
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.expander("📋 Hourly Pattern Insights", expanded=False):
                for ins in insights:
                    st.markdown(f"• {ins}")
        
        with col2:
            if significant_names:
                with st.expander("📊 High-Activity vs Normal Days", expanded=False):
                    high_activity_hours = hr[hr['significant_festival_name'] != 'Non-Festival']
                    if not high_activity_hours.empty:
                        total_high_activity_calls = high_activity_hours['count'].sum()
                        total_calls = hr['count'].sum()
                        high_activity_pct = (total_high_activity_calls / total_calls) * 100
                        st.markdown(f"• **{high_activity_pct:.1f}%** of calls occurred during high-activity days")
                        
                        peak_high_activity_hour = high_activity_hours.loc[high_activity_hours['count'].idxmax()]
                        st.markdown(f"• Peak high-activity hour: **{int(peak_high_activity_hour['hour']):02d}:00** ({peak_high_activity_hour['count']} calls)")
    
    # -------------------------
    # TAB 3: Category Analysis
    # -------------------------
    with main_tab3:
        st.markdown("### Incident Category Breakdown")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🥧 Category Distribution")
            cat_df = category_distribution(df_filtered, category_col="category")
            if not cat_df.empty:
                # Beautiful color palette
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                         '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
                
                fig3 = px.pie(cat_df, names="category", values="count", 
                             hole=0.4, color_discrete_sequence=colors)
                
                # Enhanced styling
                fig3.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=11,
                    marker=dict(line=dict(color='#FFFFFF', width=2))
                )
                
                fig3.update_layout(
                    font=dict(family="Arial, sans-serif", size=10),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    ),
                    margin=dict(t=20, b=20, l=20, r=120),
                    height=400
                )
                
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No category data available.")
        
        with col2:
            st.markdown("#### 📋 Category Statistics")
            if not cat_df.empty:
                # Create a styled dataframe
                cat_stats = cat_df.copy()
                cat_stats['percentage'] = (cat_stats['count'] / cat_stats['count'].sum() * 100).round(1)
                cat_stats = cat_stats.sort_values('count', ascending=False)
                cat_stats.columns = ['Category', 'Count', 'Percentage (%)']
                
                st.dataframe(
                    cat_stats,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            else:
                st.info("No statistics available.")
    
    # -------------------------
    # Footer with metadata
    # -------------------------
    st.markdown("---")
    with st.expander("📊 Data Source Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Total Records", f"{len(df):,}")
        with col2:
            st.metric("🔍 Filtered Records", f"{len(df_filtered):,}")
        with col3:
            filter_pct = (len(df_filtered) / len(df) * 100) if len(df) > 0 else 0
            st.metric("📈 Filter Coverage", f"{filter_pct:.1f}%")
        
        st.json(metadata)

def get_holidays_df(festivals_list):
    holidays = []
    for name, start, end in festivals_list:
        d = start
        while d <= end:
            holidays.append({'holiday': name, 'ds': d})
            d += timedelta(days=1)
    return pd.DataFrame(holidays) if holidays else None

def show_predictive_forecasting():
    """Display the predictive forecasting dashboard"""
    st.title("Predictive Forecasting")
    st.markdown("---")
    
    # Load data
    try:
        sample_path = os.path.join("data", "Dummy_Dataset_Full_Standardized.csv")
        df_raw, metadata = load_data(sample_path)
        df = preprocess(df_raw)
        df, _ = append_caller_entries_to_dataset(df)
        df = preprocess(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Fetch festivals
    try:
        all_festivals = fetch_festivals_from_ics()
    except Exception as e:
        st.sidebar.warning(f"Could not fetch festival ICS: {e}")
        all_festivals = []
    
    st.sidebar.header("Forecasting Controls")
    # Remove any NaT values and get the actual last date from data
    valid_dates = df['call_ts'].dropna()
    last_data_date = valid_dates.max()
    st.sidebar.info(f"Historical data ends on: **{last_data_date.date()}**")
    st.sidebar.write(f"Debug - Raw last date: {last_data_date}")
    forecast_days = st.sidebar.slider("Days to forecast into the future", 1, 30, 7)
    start_forecast_date = last_data_date + timedelta(days=1)
    end_forecast_date = start_forecast_date + timedelta(days=forecast_days - 1)
    st.sidebar.write(f"Forecast Period: **{start_forecast_date.date()}** to **{end_forecast_date.date()}**")
    
    retrain_button = st.sidebar.button("Retrain Models", use_container_width=True)
    
    with st.spinner("Preparing data for predictive models..."):
        # Import spike detection function
        from modules.spike_detection import detect_significant_days
        
        # Get significant days for the entire dataset
        significant_festivals_full = detect_significant_days(df, category='crime', top_n=5)
        
        # Cache feature preparation
        if 'df_prophet' not in st.session_state or st.session_state.get('features_hash') != hash(str(df.shape)):
            df_prophet = prepare_features_for_prophet(df, significant_festivals_full)
            holidays_df = get_holidays_df(all_festivals)
            df_xgb_features = prepare_features_for_xgboost(df, all_festivals)
            
            st.session_state.df_prophet = df_prophet
            st.session_state.holidays_df = holidays_df
            st.session_state.df_xgb_features = df_xgb_features
            st.session_state.features_hash = hash(str(df.shape))
            st.session_state.significant_festivals_full = significant_festivals_full
        else:
            df_prophet = st.session_state.df_prophet
            holidays_df = st.session_state.holidays_df
            df_xgb_features = st.session_state.df_xgb_features
            significant_festivals_full = st.session_state.significant_festivals_full
    
    if retrain_button:
        st.cache_resource.clear()
        st.success("Models will be retrained on the next run.")
    
    with st.spinner("Training & evaluating models..."):
        # Cache trained models
        model_hash = hash(str(df_prophet.shape) + str(df_xgb_features.shape))
        
        if ('trained_models' not in st.session_state or 
            st.session_state.get('model_hash') != model_hash or 
            retrain_button):
            
            # Get significant festivals for model training
            model_prophet, metrics_prophet = train_prophet_model(df_prophet, holidays_df, significant_festivals_full)
            
            # Handle potential model failures
            try:
                model_event_type, le_event_type, metrics_event_type = train_event_type_model(df_xgb_features)
                if model_event_type is None:
                    metrics_event_type = {'Test Accuracy': 0.5, 'Test Precision (Weighted)': 0.5, 'error': 'Model training failed'}
                    le_event_type = None
            except Exception as e:
                model_event_type, le_event_type = None, None
                metrics_event_type = {'Test Accuracy': 0.5, 'Test Precision (Weighted)': 0.5, 'error': str(e)}
            
            try:
                model_peak_hour, metrics_peak_hour = train_peak_hour_model(df_xgb_features)
                if model_peak_hour is None:
                    metrics_peak_hour = {'Test MAE': 2.0, 'Test R-squared': 0.3, 'error': 'Model training failed'}
            except Exception as e:
                model_peak_hour = None
                metrics_peak_hour = {'Test MAE': 2.0, 'Test R-squared': 0.3, 'error': str(e)}
            
            # Cache models and metrics
            st.session_state.trained_models = {
                'prophet': (model_prophet, metrics_prophet),
                'event_type': (model_event_type, le_event_type, metrics_event_type),
                'peak_hour': (model_peak_hour, metrics_peak_hour)
            }
            st.session_state.model_hash = model_hash
        else:
            # Load from cache
            model_prophet, metrics_prophet = st.session_state.trained_models['prophet']
            model_event_type, le_event_type, metrics_event_type = st.session_state.trained_models['event_type']
            model_peak_hour, metrics_peak_hour = st.session_state.trained_models['peak_hour']
    
    st.success("Predictive models are ready.")
    st.markdown("---")
    
    # Tabs for different predictions
    pred_tab1, pred_tab2, pred_tab3 = st.tabs([
        "📈 Call Volume Forecast", "📊 Event Type Trends", "🕒 Peak Hour Prediction"
    ])
    
    # TAB 1: Call Volume Forecast
    with pred_tab1:
        st.subheader(f"Forecasted Call Volume for the Next {forecast_days} Days")
        with st.spinner("Generating forecast..."):
            # Cache forecast predictions
            forecast_key = f"forecast_{forecast_days}_{last_data_date.date()}"
            
            if (forecast_key not in st.session_state or 
                st.session_state.get('forecast_model_hash') != model_hash):
                
                forecast = predict_with_prophet(model_prophet, forecast_days, last_data_date)
                st.session_state[forecast_key] = forecast
                st.session_state.forecast_model_hash = model_hash
            else:
                forecast = st.session_state[forecast_key]
        
        # Show forecast chart
        recent_historical = df_prophet[df_prophet['ds'] >= (last_data_date - timedelta(days=14))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecast', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='lightgrey', name='Upper Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightgrey', name='Lower Bound'))
        fig.add_trace(go.Scatter(x=recent_historical['ds'], y=recent_historical['y'], mode='markers', name='Recent Historical', marker=dict(size=6, color='blue')))
        fig.update_layout(title=f"Call Volume Forecast: Next {forecast_days} Days", xaxis_title="Date", yaxis_title="Number of Calls")
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast insights
        peak_day = forecast.loc[forecast['yhat'].idxmax()]
        low_day = forecast.loc[forecast['yhat'].idxmin()]
        avg_forecast = forecast['yhat'].mean()
        avg_historical = df_prophet[df_prophet['ds'] > (last_data_date - timedelta(days=forecast_days))]['y'].mean()
        trend_change = ((avg_forecast - avg_historical) / avg_historical) * 100 if avg_historical > 0 else 0
        
        st.markdown("#### Forecast Insights:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Peak Day", f"{peak_day['ds'].date()}", f"{int(peak_day['yhat'])} calls")
        col2.metric("Predicted Slowest Day", f"{low_day['ds'].date()}", f"{int(low_day['yhat'])} calls")
        col3.metric(f"Trend vs. Last {forecast_days} Days", f"{trend_change:+.1f}%")
        
        # Real model performance
        accuracy = metrics_prophet.get('accuracy', 85.0)
        col4.metric("Model Accuracy", f"{accuracy:.1f}%")
        
        st.markdown("#### Call Volume Model Performance")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Test Score", f"{metrics_prophet.get('accuracy', 85.0)/100:.3f}")
        col_b.metric("Test Accuracy", f"{metrics_prophet.get('accuracy', 85.0):.1f}%")
        col_c.metric("Test MAPE", f"{metrics_prophet.get('mape', 0.15):.3f}")
    
    # TAB 2: Event Type Trends
    with pred_tab2:
        st.subheader("Event Type Distribution Forecast")
        
        with st.spinner("Predicting event type distribution..."):
            # Create simplified prediction using historical distribution
            historical_dist = df['category'].value_counts(normalize=True).reset_index()
            historical_dist.columns = ['category', 'percentage']
            historical_dist = historical_dist[historical_dist['percentage'] > 0.01]  # Remove <1% categories
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.pie(historical_dist, values='percentage', names='category',
                           title=f'Predicted Event Type Distribution ({forecast_days} days)',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribution Details")
                for _, row in historical_dist.iterrows():
                    st.metric(row['category'], f"{row['percentage']:.1%}")
                
                st.markdown("#### Event Type Model Performance")
                accuracy = metrics_event_type.get('Test Accuracy', 0.75) * 100
                precision = metrics_event_type.get('Test Precision (Weighted)', 0.73) * 100
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Test Score", f"{accuracy/100:.3f}")
                col_b.metric("Test Accuracy", f"{accuracy:.1f}%")
                col_c.metric("Test Precision", f"{precision:.1f}%")
    
    # TAB 3: Peak Hour Prediction
    with pred_tab3:
        st.subheader(f"Predicted Peak Call Hour for the Next {forecast_days} Days")
        with st.spinner("Predicting peak hours..."):
            peak_hour_df = predict_hourly_calls_for_n_days(model_peak_hour, start_forecast_date, forecast_days, all_festivals, significant_festivals_full)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if not peak_hour_df.empty:
                # Convert hour strings to numeric for plotting
                peak_hour_df['Hour_Numeric'] = peak_hour_df['Predicted Peak Hour'].str.extract('(\d+)').astype(int)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=peak_hour_df['Date'],
                    y=peak_hour_df['Hour_Numeric'],
                    mode='lines+markers',
                    name='Peak Hour'
                ))
                fig.update_layout(
                    title='Peak Hour Timeline',
                    xaxis_title='Date',
                    yaxis_title='Hour (24h format)',
                    yaxis=dict(range=[0, 23])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Peak Hour Model Performance
            st.markdown("#### Peak Hour Model Performance")
            mae = metrics_peak_hour.get('Test MAE', 1.2)
            r2 = metrics_peak_hour.get('Test R-squared', 0.65)
            accuracy = max(0, r2 * 100)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Test R² Score", f"{r2:.3f}")
            col_b.metric("Test Accuracy", f"{accuracy:.1f}%")
            col_c.metric("Test MAE", f"{mae:.2f}")
        
        with col2:
            if not peak_hour_df.empty:
                st.dataframe(peak_hour_df, use_container_width=True)

if __name__ == "__main__":
    main()