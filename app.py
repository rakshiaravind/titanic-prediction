import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
  }

  /* Dark nautical theme */
  .stApp {
    background: linear-gradient(160deg, #0a1628 0%, #0d2240 40%, #0a1628 100%);
    color: #e8dcc8;
  }

  h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
  }

  /* Hero header */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(200,170,110,0.25);
    margin-bottom: 2rem;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #f0e0b0;
    letter-spacing: 0.04em;
    text-shadow: 0 2px 24px rgba(200,160,60,0.3);
    margin: 0;
  }
  .hero-sub {
    font-size: 1rem;
    color: #8aaccc;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(8, 20, 40, 0.97) !important;
    border-right: 1px solid rgba(200,170,110,0.15);
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label {
    color: #c8aa6e !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-size: 0.82rem;
    text-transform: uppercase;
  }

  /* Result cards */
  .result-survived {
    background: linear-gradient(135deg, rgba(30,90,60,0.6), rgba(20,60,40,0.8));
    border: 1px solid rgba(80,200,120,0.4);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .result-died {
    background: linear-gradient(135deg, rgba(90,20,20,0.6), rgba(60,10,10,0.8));
    border: 1px solid rgba(200,60,60,0.4);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .result-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 0.5rem;
  }
  .result-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
  }
  .result-prob {
    font-size: 1rem;
    opacity: 0.75;
    margin-top: 0.4rem;
  }

  /* Metric cards */
  .metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(200,170,110,0.18);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #c8aa6e;
    font-weight: 700;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #7a98b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
  }

  /* Section headers */
  .section-label {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6a88a8;
    margin-bottom: 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(200,170,110,0.12);
  }

  /* Predict button */
  .stButton > button {
    background: linear-gradient(135deg, #b8860b, #c8aa6e) !important;
    color: #0a1628 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(200,170,110,0.25) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(200,170,110,0.4) !important;
  }

  /* Dividers */
  hr { border-color: rgba(200,170,110,0.15) !important; }

  /* Streamlit overrides */
  .stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border-color: rgba(200,170,110,0.2) !important; }
  .stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

  /* Feature importance bar */
  .fi-bar-bg { background: rgba(255,255,255,0.06); border-radius: 4px; height: 8px; margin: 4px 0 10px; }
  .fi-bar { background: linear-gradient(90deg, #c8aa6e, #f0e0b0); border-radius: 4px; height: 8px; }
  .fi-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: #8aaccc; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/titanic_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def get_title_code(title_str):
    mapping = {"Mr": 3, "Miss": 2, "Mrs": 4, "Master": 1, "Rare": 0}
    return mapping.get(title_str, 3)

def encode_inputs(pclass, sex, age, fare, embarked, sibsp, parch, title_str):
    sex_enc = 1 if sex == "Male" else 0
    embarked_enc = {"Cherbourg (C)": 0, "Queenstown (Q)": 2, "Southampton (S)": 1}[embarked]
    title_enc = get_title_code(title_str)
    age_band = min(int(age / 16), 4)
    if fare < 7.91: fare_band = 0
    elif fare < 14.454: fare_band = 1
    elif fare < 31.0: fare_band = 2
    else: fare_band = 3
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    return pd.DataFrame([[pclass, sex_enc, age_band, fare_band, embarked_enc,
                          family_size, is_alone, title_enc, sibsp, parch]],
                        columns=["Pclass", "Sex", "AgeBand", "FareBand", "Embarked",
                                 "FamilySize", "IsAlone", "Title", "SibSp", "Parch"])

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-sub">April 15, 1912 · North Atlantic</p>
  <h1 class="hero-title">🚢 Titanic Survival Predictor</h1>
  <p class="hero-sub">Random Forest · Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Model check ─────────────────────────────────────────────────────────────
bundle = load_model()
if bundle is None:
    st.error("⚠️ Model not found. Run `python train_model.py` first to train and save the model.")
    st.info("Make sure `data/train.csv` (Titanic dataset) is present, then run the training script.")
    st.stop()

model: RandomForestClassifier = bundle["model"]
features: list = bundle["features"]
accuracy: float = bundle["accuracy"]

# ─── Sidebar inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Passenger Profile</div>', unsafe_allow_html=True)

    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", 1, 80, 28)

    st.markdown('<div class="section-label">Ticket & Cabin</div>', unsafe_allow_html=True)
    pclass = st.selectbox("Passenger Class", [1, 2, 3],
                          format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x])
    fare = st.slider("Fare Paid (£)", 0, 512, 32)
    embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

    st.markdown('<div class="section-label">Family Aboard</div>', unsafe_allow_html=True)
    sibsp = st.slider("Siblings / Spouses", 0, 8, 0)
    parch = st.slider("Parents / Children", 0, 6, 0)

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Survival")

# ─── Main content ─────────────────────────────────────────────────────────────
col_result, col_stats = st.columns([1.1, 1], gap="large")

with col_result:
    if predict_btn or True:  # Show placeholder on load
        title = "Mr" if sex == "Male" else "Miss"
        X = encode_inputs(pclass, sex, age, fare, embarked, sibsp, parch, title)
        proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        survival_prob = proba[1]
        death_prob = proba[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-survived">
              <span class="result-icon">🟢</span>
              <p class="result-verdict" style="color:#5de89e;">SURVIVED</p>
              <p class="result-prob">Survival probability: <b>{survival_prob:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-died">
              <span class="result-icon">🔴</span>
              <p class="result-verdict" style="color:#e85d5d;">DID NOT SURVIVE</p>
              <p class="result-prob">Survival probability: <b>{survival_prob:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(survival_prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": "#f0e0b0", "family": "Playfair Display"}},
            title={"text": "Survival Probability", "font": {"size": 13, "color": "#8aaccc", "family": "Source Sans 3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8aaccc", "tickfont": {"color": "#8aaccc"}},
                "bar": {"color": "#c8aa6e" if survival_prob >= 0.5 else "#e85d5d"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "bordercolor": "rgba(200,170,110,0.2)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(200,50,50,0.15)"},
                    {"range": [40, 60], "color": "rgba(200,170,110,0.1)"},
                    {"range": [60, 100], "color": "rgba(50,200,100,0.15)"},
                ],
                "threshold": {"line": {"color": "#f0e0b0", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8dcc8",
            height=240,
            margin=dict(t=40, b=10, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        family_size = sibsp + parch + 1
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-value">{pclass}</div>
            <div class="metric-label">Class</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{age}</div>
            <div class="metric-label">Age</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{family_size}</div>
            <div class="metric-label">Family</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">£{fare}</div>
            <div class="metric-label">Fare</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with col_stats:
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{accuracy:.1%}</div>
        <div class="metric-label">Test Accuracy</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">200</div>
        <div class="metric-label">Trees</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importances
    st.markdown('<div class="section-label">Feature Importances</div>', unsafe_allow_html=True)
    fi = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": features, "Importance": fi}).sort_values("Importance", ascending=False)

    label_map = {
        "Pclass": "Passenger Class", "Sex": "Sex", "AgeBand": "Age Group",
        "FareBand": "Fare Band", "Embarked": "Port", "FamilySize": "Family Size",
        "IsAlone": "Travelling Alone", "Title": "Title", "SibSp": "Siblings/Spouses",
        "Parch": "Parents/Children"
    }

    bars_html = ""
    for _, row in fi_df.iterrows():
        pct = row["Importance"] * 100
        label = label_map.get(row["Feature"], row["Feature"])
        bars_html += f"""
        <div class="fi-label"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="fi-bar-bg"><div class="fi-bar" style="width:{pct:.1f}%"></div></div>
        """
    st.markdown(bars_html, unsafe_allow_html=True)

    # Survival rate context chart
    st.markdown('<div class="section-label" style="margin-top:1rem;">Historical Context</div>', unsafe_allow_html=True)
    context_data = pd.DataFrame({
        "Group": ["1st Class", "2nd Class", "3rd Class", "Women", "Men", "Children"],
        "Survival %": [63, 47, 24, 74, 19, 52],
        "Color": ["#c8aa6e", "#a08040", "#7a5c20", "#5de89e", "#e85d5d", "#8aaccc"]
    })
    fig2 = px.bar(context_data, x="Group", y="Survival %",
                  color="Group", color_discrete_sequence=context_data["Color"].tolist(),
                  text="Survival %")
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8dcc8",
        font_family="Source Sans 3",
        showlegend=False,
        height=260,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, tickcolor="#6a88a8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,170,110,0.1)", range=[0, 100]),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#4a6888;font-size:0.78rem;letter-spacing:0.08em;">
  RANDOM FOREST CLASSIFIER · TRAINED ON KAGGLE TITANIC DATASET · 891 PASSENGERS
</p>
""", unsafe_allow_html=True)
