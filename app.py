import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
DATA_PATH = ROOT / "dataset" / "reviews.json"

st.set_page_config(
    page_title="ReviewSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#F6F8FB;}
[data-testid="stSidebar"] {background:#142238;}
[data-testid="stSidebar"] * {color:#F2F6FC !important;}
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1450px;}
.hero {background:linear-gradient(115deg,#142238,#203D55);padding:28px 32px;border-radius:18px;color:white;margin-bottom:22px;}
.hero h1 {color:white;font-size:2.1rem;margin:0;}
.hero p {color:#D6E3ED;margin:8px 0 0 0;}
.metric-card {background:white;border:1px solid #E7ECF2;border-radius:15px;padding:18px 20px;min-height:112px;}
.metric-label {color:#68778A;font-size:.85rem;font-weight:600;}
.metric-value {color:#172B43;font-size:1.8rem;font-weight:800;margin-top:5px;}
.result-card {background:white;border:1px solid #E7ECF2;border-radius:16px;padding:24px;margin:12px 0;}
.small-muted {color:#6B7888;font-size:.9rem;}
div.stButton > button {border-radius:10px;font-weight:700;min-height:2.8rem;}
[data-testid="stFileUploader"] {background:white;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def load_reviews():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(DATA_PATH)
    except Exception:
        return pd.DataFrame()

def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )

model = load_model()
metrics = {}
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

with st.sidebar:
    st.markdown("## 🧠 ReviewSense AI")
    st.caption("CUSTOMER REVIEW INTELLIGENCE")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Analyze a review", "Batch analysis", "Model performance", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    if model:
        st.success("Prediction engine ready")
    else:
        st.warning("Model not trained yet")
    st.caption("NLP • TF-IDF • Logistic Regression")

if page == "Overview":
    hero("Understand every customer voice.", "Turn unstructured customer feedback into clear sentiment insights.")
    df = load_reviews()
    if not df.empty:
        counts = df["sentiment"].value_counts()
        total = len(df)
        positive = int(counts.get("positive", 0))
        neutral = int(counts.get("neutral", 0))
        negative = int(counts.get("negative", 0))
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("REVIEWS IN DATASET", f"{total:,}")
        with c2: metric_card("POSITIVE", f"{positive:,}")
        with c3: metric_card("NEUTRAL", f"{neutral:,}")
        with c4: metric_card("NEGATIVE", f"{negative:,}")
        st.write("")
        left,right = st.columns([1.1, .9])
        with left:
            st.subheader("Sentiment distribution")
            order = ["positive","neutral","negative"]
            chart = counts.reindex(order, fill_value=0).rename_axis("Sentiment").reset_index(name="Reviews")
            fig = px.bar(chart, x="Sentiment", y="Reviews", color="Sentiment",
                         color_discrete_map={"positive":"#18A477","neutral":"#E5A63B","negative":"#E16B70"})
            fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",showlegend=False,
                              margin=dict(l=10,r=10,t=20,b=10),font_color="#26384D")
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.subheader("What this dataset contains")
            st.markdown("""
            - **Source:** Yelp Review Full
            - **Task:** Three-class sentiment classification
            - **Labels:** Positive, neutral and negative
            - **Input:** English-language customer reviews
            """)
            st.info("Dataset counts describe the local sample. They are not live customer activity.")
    else:
        st.info("Add dataset/reviews.json to display your dataset overview.")
    st.subheader("Quick start")
    a,b,c = st.columns(3)
    with a: st.markdown("**01 · Train**\n\nRun `python src/train.py`.")
    with b: st.markdown("**02 · Predict**\n\nAnalyze one review or upload a file.")
    with c: st.markdown("**03 · Learn**\n\nReview model metrics and limitations.")

elif page == "Analyze a review":
    hero("Analyze a customer review", "Enter feedback and let the trained NLP model classify its sentiment.")
    if model is None:
        st.error("The model is not available. Train it first using `python src/train.py`.")
    else:
        review = st.text_area("CUSTOMER REVIEW", height=190, placeholder="Paste a customer review here...")
        if st.button("Analyze sentiment", type="primary", use_container_width=True):
            if not review.strip():
                st.warning("Please enter a review first.")
            else:
                prediction = model.predict([review])[0]
                probabilities = model.predict_proba([review])[0]
                classes = list(model.classes_)
                confidence = float(max(probabilities))
                color = {"positive":"#18A477","neutral":"#D69A2C","negative":"#D95F66"}.get(prediction,"#456")
                st.markdown(
                    f'<div class="result-card"><div class="small-muted">PREDICTED SENTIMENT</div>'
                    f'<h2 style="color:{color};text-transform:capitalize;margin:8px 0">{prediction}</h2>'
                    f'<div class="small-muted">Model confidence: {confidence:.1%}</div></div>',
                    unsafe_allow_html=True
                )
                prob_df = pd.DataFrame({"Sentiment":classes,"Probability":probabilities})
                fig = px.bar(prob_df,x="Sentiment",y="Probability",color="Sentiment",
                             color_discrete_map={"positive":"#18A477","neutral":"#E5A63B","negative":"#E16B70"})
                fig.update_layout(yaxis_tickformat=".0%",yaxis_range=[0,1],showlegend=False,
                                  plot_bgcolor="white",paper_bgcolor="white")
                st.plotly_chart(fig,use_container_width=True)
                st.caption("Confidence is the model's estimated class probability, not a guarantee of correctness.")

elif page == "Batch analysis":
    hero("Batch review analysis", "Upload a JSON or CSV file containing a review column.")
    if model is None:
        st.error("Train the model before running batch predictions.")
    else:
        st.markdown("Your file must contain a column named **review**. JSON should be a list of objects.")
        uploaded = st.file_uploader("Upload reviews", type=["json","csv"])
        if uploaded:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    batch = pd.read_csv(uploaded)
                else:
                    batch = pd.read_json(uploaded)
                if "review" not in batch.columns:
                    st.error("The uploaded file must contain a 'review' column.")
                else:
                    batch = batch.dropna(subset=["review"]).copy()
                    batch["review"] = batch["review"].astype(str)
                    batch["predicted_sentiment"] = model.predict(batch["review"].tolist())
                    probabilities = model.predict_proba(batch["review"].tolist())
                    batch["confidence"] = probabilities.max(axis=1)
                    st.success(f"Analyzed {len(batch):,} reviews.")
                    st.dataframe(batch,use_container_width=True,hide_index=True)
                    counts = batch["predicted_sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
                    fig = px.pie(counts,names="Sentiment",values="Reviews",hole=.55,
                                 color="Sentiment",color_discrete_map={"positive":"#18A477","neutral":"#E5A63B","negative":"#E16B70"})
                    st.plotly_chart(fig,use_container_width=True)
                    st.download_button("Download predictions as CSV",
                        batch.to_csv(index=False).encode("utf-8"),
                        file_name="reviewsense_predictions.csv",mime="text/csv",
                        use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")

elif page == "Model performance":
    hero("Model performance", "Evaluation results from the held-out test set.")
    if not metrics:
        st.info("No evaluation report found. Run `python src/train.py` first.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1: metric_card("ACCURACY", f'{metrics.get("accuracy",0):.1%}')
        with c2: metric_card("MACRO F1", f'{metrics.get("macro_f1",0):.1%}')
        with c3: metric_card("TEST REVIEWS", f'{metrics.get("test_samples",0):,}')
        st.write("")
        st.subheader("Classification report")
        report = metrics.get("classification_report", {})
        if report:
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.style.format(precision=3),use_container_width=True)
        st.caption("The test set is held out from model training. Metrics reflect this sampled Yelp dataset, not guaranteed performance on every real-world domain.")

elif page == "About":
    hero("About ReviewSense AI", "A compact NLP project for understanding customer feedback.")
    st.subheader("How it works")
    st.markdown("""
    1. Review text is normalized using a lightweight preprocessing function.
    2. TF-IDF converts text into numerical features.
    3. Logistic Regression predicts positive, neutral or negative sentiment.
    4. The application displays predictions and estimated class probabilities.

    **Dataset labels:** Yelp ratings 1–2 are mapped to negative, 3 to neutral, and 4–5 to positive.

    **Limitations:** Sentiment is subjective. Sarcasm, mixed opinions, domain-specific language and reviews unlike Yelp data may reduce accuracy. Do not treat predictions as definitive judgments.
    """)
    st.subheader("Technology")
    st.write("Python · Pandas · Scikit-learn · NLTK-compatible text workflow · Streamlit · Plotly")
