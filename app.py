import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ======================
# LOAD MODEL
# ======================
# Ensure the model file is in the same directory
model = joblib.load("youtube_ad_revenue_model.pkl")

st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    layout="centered"
)

st.title("📺 YouTube Ad Revenue Predictor")
st.write("Enter realistic video metrics to estimate ad revenue.")

# ======================
# USER INPUTS
# ======================

with st.form("prediction_form"):
    
    # --- Date Input (Crucial for preprocessing) ---
    st.subheader("1. Date Information")
    upload_date = st.date_input(
        "Video Upload Date", 
        value=datetime.date.today(),
        help="The model uses the year, month, and day of the week as features."
    )

    # --- Categorical Inputs ---
    st.subheader("2. Video Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category = st.selectbox(
            "Category",
            ["Entertainment", "Gaming", "Education", "Music", "Tech", "Lifestyle"]
        )
    with col2:
        device = st.selectbox(
            "Device",
            ["Mobile", "Desktop", "TV", "Tablet"] # Added Tablet to match dataset
        )
    with col3:
        country = st.selectbox(
            "Country",
            ["US", "India", "UK", "Canada", "Germany", "Australia"]
        )

    video_length = st.number_input(
        "Video Length (minutes)",
        min_value=0.5,
        max_value=120.0,
        value=12.0,
        step=0.5
    )

    # --- Numerical Metrics ---
    st.subheader("3. Engagement Metrics")
    
    col4, col5 = st.columns(2)
    with col4:
        views = st.number_input(
            "Views",
            min_value=100,
            max_value=200_000_000,
            value=10_000,
            step=1000
        )
        subscribers = st.number_input(
            "Subscribers",
            min_value=0,
            max_value=100_000_000,
            value=500_000,
            step=1000
        )

    with col5:
        likes = st.number_input("Likes", min_value=0, value=500, step=10)
        comments = st.number_input("Comments", min_value=0, value=50, step=10)

    watch_time_per_view = st.slider(
        "Avg Watch Time per View (minutes)",
        min_value=0.1,
        max_value=float(video_length), # Cannot watch longer than the video
        value=3.0,
        step=0.1,
        help="Average time a user spends watching this video."
    )

    submit_btn = st.form_submit_button("Predict Revenue")

# ======================
# PREPROCESSING & INFERENCE
# ======================
if submit_btn:
    # 1. Feature Engineering: Date Components
    # The notebook extracts these specifically using .dt accessor
    year = upload_date.year
    month = upload_date.month
    dayofweek = upload_date.weekday()  # Monday=0, Sunday=6 (Matches pandas .dt.dayofweek)

    # 2. Feature Engineering: Log Transformations
    # The notebook applies np.log1p to 'views' and 'subscribers'
    log_views = np.log1p(views)
    log_subscribers = np.log1p(subscribers)

    # 3. Feature Engineering: Engagement Rate
    # Calculated as (likes + comments) / views
    # Avoid division by zero
    safe_views = views if views > 0 else 1
    engagement_rate = (likes + comments) / safe_views

    # Construct DataFrame with exact column names expected by the pipeline
    input_data = pd.DataFrame([{
        'log_views': log_views,
        'log_subscribers': log_subscribers,
        'watch_time_per_view': watch_time_per_view,
        'engagement_rate': engagement_rate,
        'video_length_minutes': video_length,
        'year': year,
        'month': month,
        'dayofweek': dayofweek,
        'category': category,
        'device': device,
        'country': country
    }])

    # Display Input Data for verification
    with st.expander("See Processed Input Data"):
        st.dataframe(input_data)

    try:
        # Prediction
        # The model output is likely log_ad_revenue_usd based on the notebook target
        log_prediction = model.predict(input_data)[0]
        
        # Reverse Log Transformation (expm1) to get actual USD
        prediction = np.expm1(log_prediction)

        st.success(f"💰 Estimated Ad Revenue: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Ensure columns match the training data exactly.")
