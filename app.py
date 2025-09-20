# app.py  (final, replace your current file with this)
import os
from datetime import timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import firebase_admin
from firebase_admin import credentials, firestore, auth
from streamlit_option_menu import option_menu

from auth_ui import initialize_session_state, signup_form, login_form, show_user_info, notification_center
from modules.data_loader import load_data, preprocess
from modules.analysis import (
    agg_calls_by_day, agg_calls_by_hour, category_distribution, compute_kpis,
    interpret_time_series, interpret_hourly_distribution
)
from modules.mapping import pydeck_points_map, pydeck_heatmap
from modules.festivals_ics import fetch_festivals_from_ics
from modules.festivals_utils import filter_significant_festivals
from modules.ui_calendar import render_month_calendar

# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

def main():
    st.set_page_config(
        page_title="Goa Police Dashboard",
        page_icon="🚔",
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
            st.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=GOA+POLICE", width=300)
            st.title("🚔 Goa Police Dashboard")
            st.markdown("**Official Portal for Goa Police Officers**")
            st.markdown("---")
        
        # Create tabs for login and signup
        tab1, tab2 = st.tabs(["🔐 Officer Login", "📝 New Registration"])
        
        with tab1:
            login_form()
        
        with tab2:
            signup_form()
    
    else:
        # User is authenticated - show main dashboard
        show_user_info()
        notification_center()
        
        st.title("112 Helpline — Analytics Dashboard")
        st.markdown("---")
        
        # -------------------------
        # Input / Load data
        # -------------------------
        st.sidebar.header("Data Input")
        uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX (call logs)", type=["csv", "xlsx"])
        use_sample = st.sidebar.checkbox("Use sample dummy data", value=True)

        df_raw, metadata = None, None
        try:
            if uploaded_file is not None:
                df_raw, metadata = load_data(uploaded_file)
                st.sidebar.success(f"Loaded uploaded file ({metadata['record_count']} rows)")
            elif use_sample:
                sample_path = os.path.join("data", "112_calls_synthetic.csv")
                if not os.path.exists(sample_path):
                    st.sidebar.error(f"Sample file not found at {sample_path}")
                else:
                    df_raw, metadata = load_data(sample_path)
                    st.sidebar.info(f"Loaded sample file ({metadata['record_count']} rows)")
            else:
                st.info("Upload a CSV/XLSX file or enable sample dataset from sidebar.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # -------------------------
        # Preprocess
        # -------------------------
        df = preprocess(df_raw)  # ensures date, hour, weekday columns exist

        # -------------------------
        # Sidebar filters (date range, category, jurisdiction)
        # -------------------------
        st.sidebar.header("Filters")
        min_date = pd.to_datetime(df["date"]).min()
        max_date = pd.to_datetime(df["date"]).max()
        date_range = st.sidebar.date_input("Date range", [min_date, max_date])

        categories = df["category"].dropna().unique().tolist()
        selected_categories = st.sidebar.multiselect("Category", options=categories, default=categories)

        jurisdictions = df["jurisdiction"].dropna().unique().tolist()
        selected_jurisdictions = st.sidebar.multiselect("Jurisdiction", options=jurisdictions, default=jurisdictions)

        # Show festival calendar right below date input (always show all festival days)
        # We'll fetch all festivals first (cached in fetch function)
        try:
            all_festivals = fetch_festivals_from_ics()
        except Exception as e:
            st.sidebar.warning(f"Could not fetch festival ICS: {e}")
            all_festivals = []

        if all_festivals:
            # build festival map for the month of the selected start date
            sel_date = pd.to_datetime(date_range[0])
            year, month = sel_date.year, sel_date.month

            festival_dates_map = {}
            for name, fs, fe in all_festivals:
                cur = pd.to_datetime(fs).date()
                endd = pd.to_datetime(fe).date()
                while cur <= endd:
                    festival_dates_map.setdefault(cur.isoformat(), []).append(name)
                    cur = cur + timedelta(days=1)

            calendar_html = render_month_calendar(year, month, festival_dates_map)
            st.sidebar.markdown("### Festivals Calendar")
            with st.sidebar:
                components.html(calendar_html, height=300)


        # -------------------------
        # Apply dataset filters to create df_filtered
        # -------------------------
        mask = (
            (pd.to_datetime(df["date"]) >= pd.to_datetime(date_range[0])) &
            (pd.to_datetime(df["date"]) <= pd.to_datetime(date_range[1])) &
            (df["category"].isin(selected_categories)) &
            (df["jurisdiction"].isin(selected_jurisdictions))
        )
        df_filtered = df[mask].copy()

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

        # --- Get the top 10 festivals by crime calls ---
        significant_festals_info = filter_significant_festivals(
            festivals_in_range_all, df, category='crime', top_n=10
        )
        # Create a set of significant festival names for quick lookup
        significant_names = {f['name'] for f in significant_festals_info}

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
        # KPIs
        # -------------------------
        kpi1, kpi2, kpi3 = st.columns(3)
        kpis = compute_kpis(df_filtered)
        kpi1.metric("Total calls (filtered)", kpis["total_calls"])
        kpi2.metric("Avg calls / day", kpis["avg_per_day"])
        kpi3.metric("Peak Call Hour", kpis["peak_hour"])

        st.markdown("---")
        left, right = st.columns([2, 1])

        # -------------------------
        # Mapping
        # -------------------------
        st.markdown("## Spatial Mapping")
        tab1, tab2= st.tabs(["Points Map", "Hotspot Heatmap"])

        with tab1:
            deck_points = pydeck_points_map(df_filtered)
            if deck_points:
                st.pydeck_chart(deck_points)
            else:
                st.info("No valid coordinates to plot.")

        with tab2:
            deck_heat = pydeck_heatmap(df_filtered)
            if deck_heat:
                st.pydeck_chart(deck_heat)
            else:
                st.info("No valid coordinates to plot heatmap.")

        # -------------------------
        # Time series (highlight significant festivals with hover-over regions)
        # -------------------------
        with left:
            st.subheader("Time Series — Calls by Day")
            ts_df = agg_calls_by_day(df_filtered, date_col="date")

            if not ts_df.empty:
                # Convert date column to datetime for proper alignment
                ts_df['date'] = pd.to_datetime(ts_df['date'])

                # Create the base line chart
                fig = px.line(ts_df, x="date", y="count", labels={"date": "Date", "count": "Calls"})
                fig.update_traces(hovertemplate='Date: %{x|%Y-%m-%d}<br>Calls: %{y}')

                if significant_festals_info:
                    y_max = ts_df['count'].max()
                    festival_dates_lookup = {name: (fs, fe) for name, fs, fe in festivals_in_range_all}

                    for info in significant_festals_info:
                        fname = info['name']
                        if fname in festival_dates_lookup:
                            fs, fe = festival_dates_lookup[fname]

                            # 1. Add the visible orange shading
                            fig.add_vrect(
                                x0=fs, x1=fe,
                                fillcolor="orange", opacity=0.15,
                                layer="below", line_width=0
                            )

                            # 2. Add an invisible bar with the correct hover template
                            hover_text = f"<b>{info['name']}</b><br>Peak Calls on Max Day: {info['max_count']}<br>Increase vs. Avg: +{info['max_pct']:.0f}%"
                            
                            fig.add_trace(go.Bar(
                                x=[fs + (fe - fs) / 2],
                                y=[y_max * 1.5],
                                width=(fe - fs).total_seconds() * 1000,
                                name=fname,
                                customdata=[hover_text],  # --- CHANGE 1: Use customdata ---
                                hovertemplate='%{customdata}<extra></extra>', # --- CHANGE 2: Reference customdata ---
                                marker_opacity=0,
                                showlegend=False
                            ))
                
                fig.update_layout(yaxis_range=[0, ts_df['count'].max() * 1.1])

                st.plotly_chart(fig, use_container_width=True)

                insights = interpret_time_series(ts_df)
                st.markdown("**Insights:**")
                for ins in insights:
                    st.markdown(f"- {ins}")
            else:
                st.info("No data for selected filters.")

            # -------------------------
            # Hourly distribution (stacked by festival_name if present)
            # -------------------------
            st.subheader("Hourly Distribution")
            # Use the 'significant_festival_name' column for the chart
            if significant_names:
                hr = df_filtered.groupby(["hour", "significant_festival_name"]).size().reset_index(name="count")
                
                # Define a specific color for the 'Non-Festival' category
                color_map = {"Non-Festival": "lightblue"}

                fig2 = px.bar(hr, x="hour", y="count", color="significant_festival_name", barmode="stack",
                            color_discrete_map=color_map,  # This line sets the color
                            labels={"hour": "Hour of Day", "count": "Calls", "significant_festival_name": "Festival"})
                
                hr_totals = hr.groupby("hour")["count"].sum().reset_index()
                insights = interpret_hourly_distribution(hr_totals)
            else:
                hr = agg_calls_by_hour(df_filtered, hour_col="hour")
                # For the non-festival case, we can ensure the default bar is also light blue
                fig2 = px.bar(hr, x="hour", y="count", labels={"hour": "Hour of Day", "count": "Calls"})
                fig2.update_traces(marker_color='skyblue') # This line sets the color
                insights = interpret_hourly_distribution(hr)

            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("**Insights:**")
            for ins in insights:
                st.markdown(f"- {ins}")

        # -------------------------
        # Category distribution (unchanged)
        # -------------------------
        with right:
            st.subheader("Category Distribution")
            cat_df = category_distribution(df_filtered, category_col="category")
            if not cat_df.empty:
                fig3 = px.pie(cat_df, names="category", values="count", title="Calls by Category", hole=0.3)
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### Data Sample")
            st.dataframe(df_filtered.head(10))

        st.markdown("---")
        st.write("Debug: data source metadata")
        st.json(metadata)
        pass

if __name__ == "__main__":
    main()
