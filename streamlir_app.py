# ============================================================
#   TOURISM EXPERIENCE ANALYTICS — Streamlit App
#   app.py — Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102,126,234,0.25);
    }
    .metric-card h2 {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .metric-card p {
        font-size: 0.85rem;
        margin: 0.3rem 0 0;
        opacity: 0.85;
        font-weight: 300;
    }
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #1a1a2e;
        border-left: 4px solid #667eea;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }
    .rec-card {
        background: #f8faff;
        border: 1px solid #e8edf5;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        border-left: 4px solid #667eea;
    }
    .rec-card h4 { margin: 0 0 0.3rem; color: #1a1a2e; font-size: 1rem; }
    .rec-card p  { margin: 0; color: #6b7280; font-size: 0.85rem; }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.4rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s;
        box-shadow: 0 4px 15px rgba(102,126,234,0.35);
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 16px;
        padding: 1.8rem;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
    .result-box h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        margin: 0.5rem 0;
    }
    .result-box p { color: #a5b4fc; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#   LOAD ALL SAVED ASSETS
# ════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    assets = {}
    try:
        assets["clf_model"]     = joblib.load("best_classification_model.pkl")
        assets["reg_model"]     = joblib.load("best_regression_model.pkl")
        assets["scaler_c"]      = joblib.load("scaler_classification.pkl")
        assets["scaler_r"]      = joblib.load("scaler_regression.pkl")
        assets["pred_ratings"]  = joblib.load("predicted_ratings.pkl")
        assets["sim_matrix"]    = joblib.load("similarity_matrix.pkl")
        assets["att_features"]  = joblib.load("attraction_features.pkl")
        assets["rec_df"]        = joblib.load("rec_df.pkl")
        assets["mat"]           = pd.read_csv("clean_master_table.csv")
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        st.info("Make sure all .pkl files and clean_master_table.csv are in the same folder as app.py")
        st.stop()
    return assets

assets      = load_assets()
clf_model   = assets["clf_model"]
reg_model   = assets["reg_model"]
scaler_c    = assets["scaler_c"]
scaler_r    = assets["scaler_r"]
pred_df     = assets["pred_ratings"]
sim_df      = assets["sim_matrix"]
att_feat    = assets["att_features"]
rec_df      = assets["rec_df"]
mat         = assets["mat"]

VISIT_MODES = {1:"Business", 2:"Couples", 3:"Family", 4:"Friends", 5:"Solo"}
MODE_EMOJI  = {1:"💼", 2:"💑", 3:"👨‍👩‍👧", 4:"👫", 5:"🧳"}
STAR_COLORS = {1:"#ef4444", 2:"#f97316", 3:"#eab308", 4:"#84cc16", 5:"#22c55e"}


# ════════════════════════════════════════════════════════════
#   SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0'>
        <div style='font-size:3rem'>🌴</div>
        <div style='font-family: Playfair Display; font-size:1.3rem;
                    font-weight:700; color:#1a1a2e'>Tourism Analytics</div>
        <div style='font-size:0.8rem; color:#6b7280; margin-top:0.3rem'>
            Powered by Machine Learning
        </div>
    </div>
    <hr style='border:none; border-top:1px solid #e8edf5; margin:0.5rem 0 1rem'>
    """, unsafe_allow_html=True)

    st.markdown("**📊 Dataset Overview**")
    st.metric("Total Transactions", f"{len(mat):,}")
    st.metric("Unique Users",       f"{mat['UserId'].nunique():,}")
    st.metric("Attractions",        f"{mat['AttractionId'].nunique()}")
    st.metric("Avg Rating",         f"{mat['Rating'].mean():.2f} ⭐")

    st.markdown("<hr style='border:none; border-top:1px solid #e8edf5'>", unsafe_allow_html=True)
    st.markdown("**🗺️ Navigate**")
    st.markdown("""
    - 🏠 Home — Overview & EDA
    - 🔮 Visit Mode Predictor
    - ⭐ Rating Predictor
    - 🎯 Recommendations
    """)


# ════════════════════════════════════════════════════════════
#   MAIN CONTENT — TABS
# ════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">Tourism Experience Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Classification · Prediction · Recommendation System</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Home",
    "🔮  Visit Mode Predictor",
    "⭐  Rating Predictor",
    "🎯  Recommendations"
])


# ────────────────────────────────────────────────────────────
# TAB 1 — HOME (EDA Dashboard)
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Dataset Snapshot</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <h2>{len(mat):,}</h2><p>Total Visits</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <h2>{mat['UserId'].nunique():,}</h2><p>Unique Users</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <h2>{mat['AttractionId'].nunique()}</h2><p>Attractions</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <h2>{mat['Rating'].mean():.2f}⭐</h2><p>Avg Rating</p></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Exploratory Analysis</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Rating Distribution
        fig, ax = plt.subplots(figsize=(6, 3.5))
        rc = mat["Rating"].value_counts().sort_index()
        bars = ax.bar(rc.index.astype(str), rc.values,
                      color=["#ef4444","#f97316","#eab308","#84cc16","#22c55e"],
                      edgecolor="white", linewidth=1.2, width=0.6)
        for bar, v in zip(bars, rc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                    f"{v:,}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title("Rating Distribution", fontweight="bold", fontsize=11)
        ax.set_xlabel("Rating"); ax.set_ylabel("Count")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Season Distribution
        fig, ax = plt.subplots(figsize=(6, 3.5))
        season_order = ["Spring","Summer","Fall","Winter"]
        sc = mat["Season"].value_counts().reindex(season_order)
        ax.bar(sc.index, sc.values,
               color=["#F18F01","#E94F37","#A23B72","#2E86AB"],
               edgecolor="white", linewidth=1.2, width=0.55)
        for i, v in enumerate(sc.values):
            ax.text(i, v+80, f"{v:,}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title("Visits by Season", fontweight="bold", fontsize=11)
        ax.set_ylabel("Count")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        # Visit Mode Pie
        fig, ax = plt.subplots(figsize=(6, 3.5))
        vc = mat[mat["VisitMode"] != "Unknown"]["VisitMode"].value_counts()
        ax.pie(vc.values, labels=vc.index,
               autopct="%1.1f%%",
               colors=["#667eea","#764ba2","#f093fb","#4facfe","#43e97b"],
               startangle=140, wedgeprops={"edgecolor":"white","linewidth":1.5})
        ax.set_title("Visit Mode Distribution", fontweight="bold", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top Attractions
        fig, ax = plt.subplots(figsize=(6, 3.5))
        top10 = mat["Attraction"].value_counts().head(8)
        colors = sns.color_palette("Blues_r", 8)
        ax.barh(range(len(top10)), top10.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels([t[:25]+"…" if len(t)>25 else t for t in top10.index], fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Top 8 Visited Attractions", fontweight="bold", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Key Insights
    st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.info("📈 **Ratings are skewed high** — 4 & 5 stars dominate, suggesting satisfied tourists or positive response bias.")
    with i2:
        st.warning("⚠️ **Class Imbalance Detected** — Couples dominate visit modes. ML models use balanced class weights to compensate.")
    with i3:
        st.success("🌴 **All attractions are in Indonesia** — Bali leads with the most visits, followed by Yogyakarta and Malang.")


# ────────────────────────────────────────────────────────────
# TAB 2 — VISIT MODE PREDICTOR
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Predict Visit Mode</p>', unsafe_allow_html=True)
    st.markdown("Fill in the details below and the model will predict how you're likely to travel.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**👤 User Profile**")
        visit_year   = st.selectbox("Visit Year",  sorted(mat["VisitYear"].dropna().unique().tolist(), reverse=True))
        visit_month  = st.selectbox("Visit Month", list(range(1, 13)),
                                    format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                           "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        user_cont    = st.selectbox("Your Continent",
                                    sorted(mat["User_Continent"].dropna().unique().tolist()))

    with col2:
        st.markdown("**🏖️ Attraction Details**")
        attraction   = st.selectbox("Attraction",
                                    sorted(mat["Attraction"].dropna().unique().tolist()))
        att_type     = st.selectbox("Attraction Type",
                                    sorted(mat["AttractionType"].dropna().unique().tolist()))
        dest_city    = st.selectbox("Destination City",
                                    sorted(mat["Dest_City"].dropna().unique().tolist()))

    st.markdown("---")
    predict_mode = st.button("🔮 Predict Visit Mode", key="pred_mode")

    if predict_mode:
        # Build feature vector matching what model was trained on
        # Use aggregate features from training data as proxies
        avg_att_rating = rec_df[rec_df["Attraction"] == attraction]["Rating"].mean()
        att_popularity = len(rec_df[rec_df["Attraction"] == attraction])

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        cont_enc  = hash(user_cont)  % 100
        att_enc   = hash(attraction) % 1000
        type_enc  = hash(att_type)   % 100
        city_enc  = hash(dest_city)  % 100

        # Season from month
        season_map = {12:"Winter",1:"Winter",2:"Winter",
                      3:"Spring",4:"Spring",5:"Spring",
                      6:"Summer",7:"Summer",8:"Summer",
                      9:"Fall",10:"Fall",11:"Fall"}
        season = season_map[visit_month]
        season_enc = {"Spring":0,"Summer":1,"Fall":2,"Winter":3}[season]

        # Build input vector
        n_features = scaler_c.n_features_in_
        feature_vec = np.zeros(n_features)

        # Fill what we can
        vals = [visit_year, visit_month, att_enc, cont_enc,
                type_enc, city_enc, avg_att_rating if not np.isnan(avg_att_rating) else 4.0,
                att_popularity, 4.0, 10, 5, season_enc]
        for i, v in enumerate(vals[:n_features]):
            feature_vec[i] = v

        feature_scaled = scaler_c.transform([feature_vec])

        try:
            pred = clf_model.predict(feature_scaled)[0]
            proba = clf_model.predict_proba(feature_scaled)[0]
        except:
            pred  = clf_model.predict([feature_vec])[0]
            proba = clf_model.predict_proba([feature_vec])[0]

        mode_name  = VISIT_MODES.get(int(pred), str(pred))
        mode_emoji = MODE_EMOJI.get(int(pred), "🧳")

        st.markdown(f"""
        <div class='result-box'>
            <p>Predicted Visit Mode</p>
            <h1>{mode_emoji}</h1>
            <h1 style='font-size:2rem; margin:0'>{mode_name}</h1>
            <p style='margin-top:0.5rem'>Confidence: {max(proba)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("**Probability Breakdown:**")
        classes = clf_model.classes_
        prob_df = pd.DataFrame({
            "Visit Mode": [VISIT_MODES.get(int(c), str(c)) for c in classes],
            "Probability": [f"{p*100:.1f}%" for p in proba],
            "Score": proba
        }).sort_values("Score", ascending=False)
        st.dataframe(prob_df[["Visit Mode","Probability"]],
                     use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────
# TAB 3 — RATING PREDICTOR
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Predict Attraction Rating</p>', unsafe_allow_html=True)
    st.markdown("Select an attraction and your profile to get a predicted rating.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏖️ Attraction**")
        r_attraction  = st.selectbox("Attraction", sorted(mat["Attraction"].dropna().unique().tolist()), key="r_att")
        r_att_type    = st.selectbox("Attraction Type", sorted(mat["AttractionType"].dropna().unique().tolist()), key="r_type")
        r_dest_city   = st.selectbox("Destination City", sorted(mat["Dest_City"].dropna().unique().tolist()), key="r_city")

    with col2:
        st.markdown("**👤 Your Profile**")
        r_visit_mode  = st.selectbox("How are you travelling?",
                                     ["Business","Couples","Family","Friends","Solo"], key="r_mode")
        r_visit_year  = st.selectbox("Year", sorted(mat["VisitYear"].dropna().unique().tolist(), reverse=True), key="r_year")
        r_visit_month = st.selectbox("Month", list(range(1,13)),
                                     format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                            "Jul","Aug","Sep","Oct","Nov","Dec"][x-1],
                                     key="r_month")
        r_user_cont   = st.selectbox("Your Continent",
                                     sorted(mat["User_Continent"].dropna().unique().tolist()), key="r_cont")

    st.markdown("---")
    predict_rating = st.button("⭐ Predict Rating", key="pred_rating")

    if predict_rating:
        # Aggregate features
        avg_att = rec_df[rec_df["Attraction"] == r_attraction]["Rating"].mean()
        att_pop = len(rec_df[rec_df["Attraction"] == r_attraction])
        mode_id = {"Business":1,"Couples":2,"Family":3,"Friends":4,"Solo":5}[r_visit_mode]

        att_enc   = hash(r_attraction) % 1000
        cont_enc  = hash(r_user_cont)  % 100
        type_enc  = hash(r_att_type)   % 100
        city_enc  = hash(r_dest_city)  % 100
        season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
        season_enc = season_map[r_visit_month]

        n_features = scaler_r.n_features_in_
        feature_vec = np.zeros(n_features)
        vals = [r_visit_year, r_visit_month, mode_id, att_enc,
                cont_enc, type_enc, city_enc,
                avg_att if not np.isnan(avg_att) else 4.0,
                att_pop, 4.0, 10, 5, season_enc]
        for i, v in enumerate(vals[:n_features]):
            feature_vec[i] = v

        try:
            feature_scaled = scaler_r.transform([feature_vec])
            pred_rating    = reg_model.predict(feature_scaled)[0]
        except:
            pred_rating    = reg_model.predict([feature_vec])[0]

        pred_rating = float(np.clip(pred_rating, 1.0, 5.0))
        stars       = "⭐" * round(pred_rating)
        color       = STAR_COLORS.get(round(pred_rating), "#667eea")

        st.markdown(f"""
        <div class='result-box'>
            <p>Predicted Rating for <b>{r_attraction}</b></p>
            <h1 style='font-size:4rem; margin:0.3rem 0'>{stars}</h1>
            <h1 style='color:{color}'>{pred_rating:.1f} / 5.0</h1>
            <p>Based on attraction features and your travel profile</p>
        </div>
        """, unsafe_allow_html=True)

        # Context stats
        st.markdown("**📊 Attraction Stats from Historical Data:**")
        s1, s2, s3 = st.columns(3)
        s1.metric("Historical Avg Rating", f"{avg_att:.2f} ⭐" if not np.isnan(avg_att) else "N/A")
        s2.metric("Total Visits",          f"{att_pop:,}")
        s3.metric("Predicted Rating",      f"{pred_rating:.2f} ⭐")


# ────────────────────────────────────────────────────────────
# TAB 4 — RECOMMENDATIONS
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Personalized Recommendations</p>', unsafe_allow_html=True)
    st.markdown("Enter a User ID to get personalized attraction recommendations.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sample_ids = rec_df["UserId"].value_counts().head(20).index.tolist()
        user_id_input = st.selectbox("Select User ID (or type one)",
                                     options=sample_ids,
                                     help="These are the most active users in the dataset")
    with col2:
        n_recs = st.slider("Number of recommendations", 3, 10, 5)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        get_recs = st.button("🎯 Get Recommendations", key="get_recs")

    if get_recs:
        # Show user history
        user_hist = rec_df[rec_df["UserId"] == user_id_input][
            ["Attraction","AttractionType","Dest_City","Rating"]
        ].sort_values("Rating", ascending=False)

        st.markdown(f'<p class="section-header">📋 Visit History — User {user_id_input}</p>',
                    unsafe_allow_html=True)
        if user_hist.empty:
            st.warning("No visit history found for this user.")
        else:
            st.dataframe(user_hist.reset_index(drop=True),
                         use_container_width=True, hide_index=True)

        rec_col1, rec_col2 = st.columns(2)

        # ── Collaborative Filtering ──────────────────────
        with rec_col1:
            st.markdown('<p class="section-header">🤝 Collaborative Filtering</p>',
                        unsafe_allow_html=True)
            st.caption("Based on what similar users enjoyed")

            if user_id_input in pred_df.index:
                visited     = rec_df[rec_df["UserId"] == user_id_input]["AttractionId"].tolist()
                user_pred   = pred_df.loc[user_id_input]
                not_visited = user_pred.drop(index=visited, errors="ignore")
                top_n_ids   = not_visited.nlargest(n_recs).index.tolist()

                recs = att_feat[att_feat["AttractionId"].isin(top_n_ids)].copy()
                recs["Predicted_Rating"] = recs["AttractionId"].map(
                    not_visited[top_n_ids].to_dict()
                ).round(2)
                recs = recs.sort_values("Predicted_Rating", ascending=False)

                for _, row in recs.iterrows():
                    stars = "⭐" * min(5, max(1, round(float(row["Predicted_Rating"]))))
                    st.markdown(f"""
                    <div class='rec-card'>
                        <h4>{row['Attraction']}</h4>
                        <p><span class='badge'>{row['AttractionType']}</span>
                           <span class='badge'>{row['Dest_City']}</span></p>
                        <p style='margin-top:0.4rem'>{stars} {row['Predicted_Rating']:.2f} predicted</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"User {user_id_input} not found in training data.")

        # ── Content-Based Filtering ──────────────────────
        with rec_col2:
            st.markdown('<p class="section-header">🏷️ Content-Based Filtering</p>',
                        unsafe_allow_html=True)
            st.caption("Based on attraction features you enjoyed")

            user_visits = rec_df[rec_df["UserId"] == user_id_input][["AttractionId","Rating"]]
            if not user_visits.empty:
                visited_ids    = user_visits["AttractionId"].tolist()
                similar_scores = pd.Series(dtype=float)

                for _, row in user_visits.iterrows():
                    att_id = row["AttractionId"]
                    rating = float(row["Rating"])
                    if att_id not in sim_df.index:
                        continue
                    sims = sim_df[att_id].astype(float) * rating
                    similar_scores = similar_scores.add(sims, fill_value=0)

                similar_scores = similar_scores.drop(index=visited_ids, errors="ignore")
                top_n_ids      = similar_scores.nlargest(n_recs).index.tolist()

                recs_cb = att_feat[att_feat["AttractionId"].isin(top_n_ids)].copy()
                recs_cb["Similarity_Score"] = recs_cb["AttractionId"].map(
                    similar_scores[top_n_ids].to_dict()
                ).round(3)
                recs_cb = recs_cb.sort_values("Similarity_Score", ascending=False)

                for _, row in recs_cb.iterrows():
                    st.markdown(f"""
                    <div class='rec-card' style='border-left-color:#764ba2'>
                        <h4>{row['Attraction']}</h4>
                        <p><span class='badge'>{row['AttractionType']}</span>
                           <span class='badge'>{row['Dest_City']}</span></p>
                        <p style='margin-top:0.4rem'>
                            Similarity Score: <b>{row['Similarity_Score']:.3f}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No visit history found for content-based recommendations.")

    else:
        # Placeholder when no button pressed yet
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#9ca3af;
                    background:#f8faff; border-radius:16px; border:2px dashed #e8edf5'>
            <div style='font-size:3rem'>🎯</div>
            <p style='font-size:1.1rem; margin:0.5rem 0'>Select a User ID and click Get Recommendations</p>
            <p style='font-size:0.85rem'>Both Collaborative and Content-Based results will appear here</p>
        </div>
        """, unsafe_allow_html=True)
