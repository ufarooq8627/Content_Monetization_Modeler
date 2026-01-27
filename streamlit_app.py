import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# LOAD MODEL
# ======================
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

views = st.number_input(
    "Views",
    min_value=1_000,
    max_value=200_000_000,
    value=1_000_000,
    step=100_000
)

subscribers = st.number_input(
    "Subscribers",
    min_value=0,
    max_value=100_000_000,
    value=500_000,
    step=10_000
)

watch_time_per_view = st.slider(
    "Watch Time per View (minutes)",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.1
)

likes = st.number_input(
    "Likes",
    min_value=0,
    max_value=int(views * 0.3),
    value=int(views * 0.05),
    step=1000
)

comments = st.number_input(
    "Comments",
    min_value=0,
    max_value=int(views * 0.05),
    value=int(views * 0.005),
    step=100
)

video_length = st.slider(
    "Video Length (minutes)",
    min_value=0.5,
    max_value=120.0,
    value=12.0,
    step=0.5
)

category = st.selectbox(
    "Category",
    ["Entertainment", "Gaming", "Education", "Music", "Tech", "Lifestyle"]
)

device = st.selectbox(
    "Device",
    ["Mobile", "Desktop", "TV"]
)

country = st.selectbox(
    "Country",
    ["US", "India", "UK", "Canada", "Germany", "Australia"]
)

# ======================
# FEATURE ENGINEERING
# ======================

engagement_rate = (likes + comments) / views
engagement_rate = min(engagement_rate, 0.3)  # clamp to realistic range

input_df = pd.DataFrame([{
    "log_views": np.log1p(views),
    "log_subscribers": np.log1p(subscribers),
    "watch_time_per_view": watch_time_per_view,  # MUST MATCH PIPELINE
    "engagement_rate": engagement_rate,
    "video_length_minutes": video_length,
    "year": 2024,
    "month": 1,
    "dayofweek": 2,
    "category": category,
    "device": device,
    "country": country
}])

# ======================
# PREDICTION
# ======================

if st.button("Predict Revenue"):
    revenue = model.predict(input_df)[0]
    revenue = max(revenue, 0)  # safety guard

    st.success(f"💰 Estimated Ad Revenue: **${revenue:,.2f}**")

    with st.expander("🔍 Debug Info"):
        st.write("Engagement Rate:", round(engagement_rate, 4))
        st.write("Model Output (USD):", revenue)
        st.write("Input DataFrame:")
        st.dataframe(input_df)
