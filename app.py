#  
# YouTube Ad Revenue Predictor – Streamlit App
#  
# Tabs:
#   1. Revenue Predictor  – user inputs → model prediction
#   2. EDA & Visual Analytics – dataset distributions & insights
#   3. Model Insights      – feature importance + evaluation metrics
#  

# 1. IMPORT REQUIRED LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

#  Page config (must be first Streamlit call) 
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="📺",
    layout="wide"
)

#  Custom CSS for a polished dark look 
st.markdown("""
<style>
    /* Overall background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1c1f2e;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 14px;
    }

    /* Section headers */
    h2, h3 { color: #c084fc; }

    /* Success box */
    div[data-testid="stAlert"] { border-radius: 10px; }

    /* Tab styling */
    button[data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


#  Matplotlib dark style helper ─
def apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor("#1c1f2e")
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor("#1c1f2e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#c084fc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3250")


#  Load model & feature config ─
@st.cache_resource
def load_artifacts():
    model = joblib.load("youtube_ad_revenue_model.pkl")
    feature_config = joblib.load("feature_config.pkl")
    return model, feature_config

model, feature_config = load_artifacts()


#  Load & cache a sample of the dataset for EDA 
@st.cache_data
def load_sample_data():
    df = pd.read_csv("youtube_ad_revenue_dataset.csv")
    # Sample 5 000 rows so charts render quickly in the app
    return df.dropna(subset=["ad_revenue_usd"]).sample(n=5000, random_state=42)

df_sample = load_sample_data()


#  Pre-compute feature importance from model ─
@st.cache_data
def get_feature_importance():
    prep = model.named_steps["preprocessor"]
    xgb = model.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    importances = xgb.feature_importances_

    # Clean up prefixes for readability
    clean_names = (
        pd.Series(feature_names)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    fi = pd.DataFrame({"feature": clean_names, "importance": importances})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)

fi_df = get_feature_importance()

# Hard-coded model evaluation metrics from the Jupyter Notebook training run
MODEL_METRICS = {
    "R² Score": 0.9921,
    "RMSE (USD)": 5.54,
    "MAE (USD)": 4.39,
}

# Comparison of the 5 models evaluated in the notebook
MODEL_COMPARISON = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest",
        "XGBoost ✅ (Best)",
    ],
    "R²": [0.9783, 0.9783, 0.9783, 0.9891, 0.9921],
    "RMSE": [9.20, 9.20, 9.20, 6.53, 5.54],
    "MAE": [7.42, 7.42, 7.42, 5.02, 4.39],
})

#  Colour palette  
PURPLE = "#c084fc"
CYAN   = "#22d3ee"
AMBER  = "#fbbf24"
GREEN  = "#34d399"
PINK   = "#f472b6"
PALETTE = [PURPLE, CYAN, AMBER, GREEN, PINK, "#f87171"]

#  
# APP HEADER
#  
st.markdown("# 📺 YouTube Ad Revenue Predictor")
st.markdown("*Predict ad revenue · Explore trends · Understand what drives earnings*")
st.divider()

#  
# TABS
#  
tab1, tab2, tab3 = st.tabs([
    "🎯  Revenue Predictor",
    "📊  EDA & Visual Analytics",
    "🧠  Model Insights",
])


# 
#   TAB 1 – REVENUE PREDICTOR                              
with tab1:
    st.subheader("Enter Video Metrics to Predict Ad Revenue")
    st.write("Fill in the details below and click **Predict Revenue** to get an estimate.")

    with st.form("prediction_form"):

        #  Date  ─
        st.markdown("#### 1. Date Information")
        upload_date = st.date_input(
            "Video Upload Date",
            value=datetime.date.today(),
            help="Year, month & day-of-week are extracted as model features.",
        )

        #  Categorical 
        st.markdown("#### 2. Video Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.selectbox(
                "Category",
                ["Entertainment", "Gaming", "Education", "Music", "Tech", "Lifestyle"],
            )
        with col2:
            device = st.selectbox("Device", ["Mobile", "Desktop", "TV", "Tablet"])
        with col3:
            country = st.selectbox(
                "Country",
                ["US", "India", "UK", "Canada", "Germany", "Australia"],
            )

        video_length = st.number_input(
            "Video Length (minutes)",
            min_value=0.5, max_value=120.0, value=12.0, step=0.5,
        )

        #  Numerical 
        st.markdown("#### 3. Engagement Metrics")
        col4, col5 = st.columns(2)
        with col4:
            views = st.number_input(
                "Views", min_value=100, max_value=200_000_000, value=10_000, step=1000,
            )
            subscribers = st.number_input(
                "Subscribers", min_value=0, max_value=100_000_000, value=500_000, step=1000,
            )
        with col5:
            likes = st.number_input("Likes", min_value=0, value=500, step=10)
            comments = st.number_input("Comments", min_value=0, value=50, step=10)

        watch_time_per_view = st.slider(
            "Avg Watch Time per View (minutes)",
            min_value=0.1,
            max_value=float(video_length),
            value=min(3.0, float(video_length)),
            step=0.1,
            help="Average time a viewer spends watching this video.",
        )

        submit_btn = st.form_submit_button("🔮 Predict Revenue", use_container_width=True)

    #  On submit  ─
    if submit_btn:
        # Feature engineering
        year       = upload_date.year
        month      = upload_date.month
        dayofweek  = upload_date.weekday()           # Mon=0, Sun=6
        log_views       = np.log1p(views)
        log_subscribers = np.log1p(subscribers)
        safe_views      = max(views, 1)
        engagement_rate = (likes + comments) / safe_views

        # Country code mapping (model was trained with codes)
        country_map = {
            "US": "US", "India": "IN", "UK": "UK",
            "Canada": "CA", "Germany": "DE", "Australia": "AU",
        }

        input_data = pd.DataFrame([{
            "log_views":            log_views,
            "log_subscribers":      log_subscribers,
            "watch_time_per_view":  watch_time_per_view,
            "engagement_rate":      engagement_rate,
            "video_length_minutes": video_length,
            "year":                 year,
            "month":                month,
            "dayofweek":            dayofweek,
            "category":             category,
            "device":               device,
            "country":              country_map[country],
        }])

        with st.expander("🔍 See Processed Input Data"):
            st.dataframe(input_data, use_container_width=True)

        try:
            log_prediction = model.predict(input_data)[0]
            prediction     = np.expm1(log_prediction)

            st.success(f"💰 Estimated Ad Revenue: **${prediction:,.2f} USD**")

            # Mini KPI breakdown
            k1, k2, k3 = st.columns(3)
            k1.metric("Engagement Rate", f"{engagement_rate:.4f}")
            k2.metric("Log Views",       f"{log_views:.2f}")
            k3.metric("Watch Time / View", f"{watch_time_per_view} min")

            # Revenue gauge bar
            st.markdown("##### Revenue estimate relative to dataset range")
            rev_min, rev_max = 126.6, 382.8       # dataset min/max
            pct = min(max((prediction - rev_min) / (rev_max - rev_min), 0), 1)
            st.progress(pct, text=f"${prediction:,.2f}  (dataset range: $127 – $383)")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Ensure the model file is present and columns match training data.")


 #    TAB 2 – EDA & VISUAL ANALYTICS                          
#  
with tab2:
    st.subheader("Exploratory Data Analysis")
    st.caption("Charts are based on a random sample of 5,000 records from the full 122,400-row dataset.")

    #  Row 1: Revenue distribution + Category revenue ─
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Revenue Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df_sample["ad_revenue_usd"], bins=40, color=PURPLE, alpha=0.85,
                edgecolor="#0f1117")
        ax.set_xlabel("Ad Revenue (USD)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Ad Revenue")
        apply_dark_style(fig, ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with r1c2:
        st.markdown("**Avg Revenue by Category**")
        cat_rev = (
            df_sample.groupby("category")["ad_revenue_usd"]
            .mean()
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(cat_rev.index, cat_rev.values,
                       color=PALETTE[:len(cat_rev)], edgecolor="#0f1117")
        ax.set_xlabel("Avg Revenue (USD)")
        ax.set_title("Revenue by Content Category")
        apply_dark_style(fig, ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    #  Row 2: Device share + Country revenue 
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**Revenue Share by Device**")
        dev_rev = df_sample.groupby("device")["ad_revenue_usd"].mean()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        wedges, texts, autotexts = ax.pie(
            dev_rev.values,
            labels=dev_rev.index,
            autopct="%1.1f%%",
            colors=PALETTE[:len(dev_rev)],
            startangle=140,
            textprops={"color": "#cccccc"},
        )
        for at in autotexts:
            at.set_color("#0f1117")
            at.set_fontweight("bold")
        ax.set_title("Device Type Revenue Share")
        apply_dark_style(fig, ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with r2c2:
        st.markdown("**Avg Revenue by Country**")
        cty_rev = (
            df_sample.groupby("country")["ad_revenue_usd"]
            .mean()
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(cty_rev.index, cty_rev.values,
               color=PALETTE[:len(cty_rev)], edgecolor="#0f1117")
        ax.set_xlabel("Country")
        ax.set_ylabel("Avg Revenue (USD)")
        ax.set_title("Revenue by Country")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        apply_dark_style(fig, ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    #  Row 3: Views vs Revenue scatter ─
    st.markdown("**Views vs Ad Revenue (scatter)**")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(
        np.log1p(df_sample["views"]),
        df_sample["ad_revenue_usd"],
        alpha=0.25, s=8, color=CYAN,
    )
    ax.set_xlabel("log(Views + 1)")
    ax.set_ylabel("Ad Revenue (USD)")
    ax.set_title("Log-Views vs Ad Revenue")
    apply_dark_style(fig, ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    #  Row 4: Watch time vs Revenue scatter ─
    st.markdown("**Watch-Time-per-View vs Ad Revenue (scatter)**")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(
        df_sample["watch_time_minutes"] / df_sample["views"].replace(0, 1),
        df_sample["ad_revenue_usd"],
        alpha=0.25, s=8, color=AMBER,
    )
    ax.set_xlabel("Watch Time per View (minutes)")
    ax.set_ylabel("Ad Revenue (USD)")
    ax.set_title("Avg Watch Time per View vs Ad Revenue")
    apply_dark_style(fig, ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    #  Insights callout 
    st.divider()
    st.markdown("#### 💡 Key EDA Insights")
    ins1, ins2, ins3 = st.columns(3)
    ins1.info("📌 **Revenue** ranges from ~$127 to ~$383. Minimal skew – no log transform needed on the target for distribution.")
    ins2.info("📌 **Category & Country** show modest revenue differences. Content type alone is not the dominant driver.")
    ins3.info("📌 **Watch-time per view** has the strongest visual correlation with ad revenue — confirmed by feature importance.")


 #    TAB 3 – MODEL INSIGHTS                                   
 
with tab3:

    #  Evaluation Metrics ─
    st.subheader("Model Evaluation Metrics")
    st.caption("Evaluated on a 20% hold-out test set.")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",     f"{MODEL_METRICS['R² Score']:.4f}",  "Excellent fit")
    m2.metric("RMSE (USD)",   f"${MODEL_METRICS['RMSE (USD)']:.2f}", "Low error")
    m3.metric("MAE (USD)",    f"${MODEL_METRICS['MAE (USD)']:.2f}",  "Low error")

    st.divider()

    #  Model Comparison Table 
    st.subheader("5-Model Comparison")
    st.caption("All five regression models were trained and evaluated; XGBoost performed best.")

    def highlight_best(row):
        """Highlight the XGBoost row."""
        if "XGBoost" in row["Model"]:
            return ["background-color: #2e1065; color: #c084fc; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled = MODEL_COMPARISON.style.apply(highlight_best, axis=1).format(
        {"R²": "{:.4f}", "RMSE": "${:.2f}", "MAE": "${:.2f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = [PURPLE if "XGBoost" in m else "#374151" for m in MODEL_COMPARISON["Model"]]
    short_names = ["LR", "Ridge", "Lasso", "RF", "XGB ✅"]

    for ax, metric, ylabel in zip(
        axes,
        ["R²", "RMSE", "MAE"],
        ["R² Score (higher ↑)", "RMSE USD (lower ↓)", "MAE USD (lower ↓)"],
    ):
        ax.bar(short_names, MODEL_COMPARISON[metric], color=colors, edgecolor="#0f1117")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        apply_dark_style(fig, ax)
        # Annotate bars
        for i, v in enumerate(MODEL_COMPARISON[metric]):
            ax.text(i, v * 0.98, f"{v:.2f}", ha="center", va="top",
                    fontsize=8, color="#f0f0f0")

    best_patch = mpatches.Patch(color=PURPLE, label="Best Model (XGBoost)")
    fig.legend(handles=[best_patch], loc="lower right", facecolor="#1c1f2e",
               labelcolor="#cccccc", fontsize=9)
    fig.tight_layout()
    apply_dark_style(fig, list(axes))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    #  Feature Importance ─
    st.subheader("Feature Importance (XGBoost)")
    st.caption("Shows how much each feature contributes to the model's predictions.")

    top_n  = st.slider("Show top N features", min_value=5, max_value=24, value=10)
    fi_top = fi_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.4)))
    bars = ax.barh(
        fi_top["feature"],
        fi_top["importance"],
        color=[PURPLE if i == len(fi_top) - 1 else CYAN for i in range(len(fi_top))],
        edgecolor="#0f1117",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances")
    apply_dark_style(fig, ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    #  Feature explanation table 
    st.markdown("#### Feature Explanation")
    feat_explain = pd.DataFrame({
        "Feature": [
            "watch_time_per_view", "engagement_rate", "log_views",
            "log_subscribers", "video_length_minutes",
            "country_*", "category_*", "device_*",
            "year / month / dayofweek",
        ],
        "Description": [
            "Avg minutes a viewer watches — STRONGEST signal for revenue",
            "(likes + comments) / views — captures audience interaction",
            "Log-transformed view count — reduces skewness impact",
            "Log-transformed subscriber count — channel authority proxy",
            "Duration of the video in minutes",
            "One-hot encoded country — e.g. DE, US, CA",
            "One-hot encoded content type — e.g. Gaming, Music",
            "One-hot encoded device — e.g. TV, Mobile",
            "Temporal features extracted from upload date",
        ],
        "Origin": [
            "Engineered", "Engineered", "Engineered",
            "Engineered", "Raw",
            "Raw + Encoded", "Raw + Encoded", "Raw + Encoded",
            "Engineered",
        ],
    })
    st.dataframe(feat_explain, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 💡 Key Model Insights")
    i1, i2, i3 = st.columns(3)
    i1.success("🏆 **Watch-time per view** is by far the most important feature (88% importance). Longer viewer retention = significantly higher revenue.")
    i2.warning("📐 **Engagement rate** is second (3.2%). Likes and comments signal content quality to YouTube's ad placement system.")
    i3.info("🌍 **Country matters** more than category or device. High CPM markets (DE, CA, US) generate noticeably higher revenue.")
