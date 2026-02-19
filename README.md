# 🌴 Tourism Experience Analytics

An end-to-end Data Science project that analyzes tourism behavior and delivers
personalized recommendations, visit mode classification, and rating prediction
through an interactive Streamlit application.

---

## 🎯 Objectives

| Task | Description | Target |
|---|---|---|
| Classification | Predict how a user will travel | Couples / Family / Friends / Solo / Business |
| Regression | Predict rating a user will give | 1 – 5 Stars |
| Recommendation | Suggest attractions to a user | Ranked attraction list |

---

## 📂 Dataset

9 relational Excel files — **52,930 transactions**, **33,530 users**, **30 attractions** across Bali, Malang, and Yogyakarta (Indonesia).

---

## 🔧 Tech Stack

`Python` · `Pandas` · `Scikit-Learn` · `XGBoost` · `Streamlit` · `Matplotlib` · `Seaborn`

---

## 🚀 Pipeline

```
Raw Data (9 files)
    → Data Merging        (9 Left Joins → Master Analytical Table)
    → Data Cleaning       (nulls, duplicates, type fixes, validation)
    → EDA                 (10 visualizations, key insights)
    → Feature Engineering (5 aggregate features + Label & One-Hot Encoding)
    → Model Building      (3 models each for Classification & Regression)
    → Recommendation      (Collaborative Filtering + Content-Based Filtering)
    → Streamlit App       (4-tab interactive dashboard)
```

---

## 📊 Models Used

**Classification** — Logistic Regression · Random Forest · XGBoost  
**Regression** — Linear Regression · Random Forest · XGBoost  
**Recommendation** — SVD (Collaborative) · Cosine Similarity (Content-Based)

---

## 🖥️ App Tabs

- 🏠 **Home** — EDA charts and dataset overview
- 🔮 **Visit Mode Predictor** — Predict travel mode with confidence score
- ⭐ **Rating Predictor** — Predict star rating for any attraction
- 🎯 **Recommendations** — Personalized attraction suggestions

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Make sure all `.pkl` files and `clean_master_table.csv`
> are in the same folder as `app.py`

---

## 📌 Key Findings

- 78% of ratings are 4–5 stars — tourists are generally satisfied
- Couples dominate visit modes (41%) — severe class imbalance
- Bali accounts for 85% of all visits
- Nature & Wildlife and Beaches are the most visited attraction types

---

## 👤 Author

**Your Name** · [LinkedIn](#) · [GitHub](#)
