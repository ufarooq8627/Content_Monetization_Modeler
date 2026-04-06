# 📊 YouTube Ad Revenue Predictor – Content Monetization Model

## 📌 Project Overview

This project builds an end-to-end **content monetization prediction system** using YouTube video analytics data. The goal is to **predict ad revenue** based on reach, engagement, viewer retention, content category, device type, and geography.

The project follows a **complete data science lifecycle**:
* Data cleaning & validation
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training & evaluation (including Random Forest, Linear Regression, etc.)
* Deployment through an interactive Streamlit App

---

## 🧠 Key Business Question

**What factors most strongly drive YouTube ad revenue, and how accurately can revenue be predicted from video performance metrics?**

---

## 🗂 Dataset

* Source: `youtube_ad_revenue_dataset.csv`
* Size: ~120,000 videos
* **Key features**:
  * Views, likes, comments (Reach and Engagement)
  * Watch time & video length (Retention)
  * Subscribers
  * Category, device, country
  * Ad revenue (USD) - Target Variable

---

## 🧹 Data Preprocessing & Feature Engineering

1. **Cleaning**: Handled logical errors (negative values), missing data (imputed with medians), and duplicates.
2. **Log Transformation**: Applied to skewed numerical data (`log_views`, `log_subscribers`, `log_ad_revenue_usd`) to stabilize model variance. 
3. **Engineering**:
   * **Engagement Rate**: `(likes + comments) / views`
   * **Watch Time per View**: `watch_time_minutes / views` (Crucial indicator of content quality vs. simple clickbait).
   * Date parts derived from Upload Date (`year`, `month`, `dayofweek`).

---

## 📈 Exploratory Data Analysis (EDA) Insights

* **Watch time per view** shows the strongest relationship with ad revenue. Quality retention beats pure scale.
* Engagement rate has a limited standalone impact.
* Specific categories, devices, and countries show distributional shifts but are secondary to retention metrics.

---

## 🏆 Modeling Approach & Performance

* **Target Variable**: `log_ad_revenue_usd`
* Evaluated models included Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, and **XGBoost**.
* **Model Choice**: **XGBoost** performed best as it properly captures the non-linear interaction between features like retention and revenue. 
* **Key Drivers**: 
  1. Watch time per view
  2. Log views
  3. Engagement rate

---

## 🚀 How to Run the Project

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib joblib streamlit xgboost
   ```

2. **Run the Analysis Notebook**:
   Explore the data processing and training pipeline by opening:
   ```bash
   jupyter notebook con_mon_modelr.ipynb
   ```

3. **Run the Streamlit Dashboard**:
   Interact with the trained inference model using real-world numbers!
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📁 Project Structure

```text
├── con_mon_modelr.ipynb             # Main Data Science Notebook (EDA & Training)
├── app.py                           # Interactive Web Application
├── youtube_ad_revenue_dataset.csv   # Raw Dataset
├── youtube_ad_revenue_model.pkl     # Trained XGBoost Pipeline
├── feature_config.pkl               # Model Metadata
└── README.md                        # Project Documentation
```

---

## 📌 Conclusion

Viewer retention (not just accidental views) is the **primary driver of monetization**. Optimizing **watch time per view** leads to more reliable and sustainable revenue growth than focusing solely on clickbait reach.

---

## 📬 Author
**Umar Farooque**
Data Science | Machine Learning | Analytics
