import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import matplotlib.pyplot as plt

import streamlit as st

# ------------------- BEAUTIFUL FULL UI CSS -------------------
def apply_full_style():
    st.markdown("""
        <style>

        /* Background */
        .stApp {
            background: linear-gradient(135deg, #f6f9fc, #e9eff5);
            color: #1a1a1a !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings */
        h1, h2, h3, h4 {
            font-weight: 700 !important;
            color: #1a2a4a !important;
        }

        /* Cards */
        .card {
            background: #ffffffcc;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-top: 10px;
            margin-bottom: 20px;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #4b79a1, #283e51);
            color: white;
            padding: 10px 22px;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, #35516b, #1c2a33);
        }

        /* Text Input & Selectbox */
        .stTextInput>div>div>input,
        .stSelectbox>div>div>select {
            border-radius: 10px;
            border: 1.5px solid #c3ccd6;
            padding: 8px;
            background-color: #ffffffee;
        }

        /* Dataframe styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Info, success, error blocks */
        .stAlert {
            border-radius: 10px;
        }

        </style>
    """, unsafe_allow_html=True)

apply_full_style()

# ------------------- BEAUTIFUL LIGHT NEWS BACKGROUND -------------------
def set_news_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f2f5f7, #dbe2e8);
        }
        </style>
    """, unsafe_allow_html=True)

set_news_background()
# -----------------------------------------------------------------------


# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI News Orchestrator", layout="wide")
st.title("📰 AI News Orchestrator")
st.markdown("### **One Topic → Multiple Sources → One Authentic Summary**")

NEWS_API_KEY = "2090742904994250b1d79117251e88ab"  


# ------------------- LOAD MODELS -------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

summarizer = load_summarizer()
embedder = load_embedder()


# ------------------- FUNCTIONS -------------------

def fetch_news(query, page_size=40):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        return pd.DataFrame()

    df = pd.DataFrame(data.get("articles", []))
    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    return df


def preprocess(df):
    df["full_text"] = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["content"].fillna("")
    )
    df = df[df["full_text"].str.len() > 100]
    df = df.drop_duplicates(subset=["title"])
    df = df.sort_values("publishedAt")
    df.reset_index(drop=True, inplace=True)
    return df


def summarize(text):
    if len(text.split()) < 40:
        return text
    try:
        return summarizer(text, max_length=120, min_length=20, do_sample=False)[0]["summary_text"]
    except:
        return text[:300]


def add_summaries(df):
    if "summary" not in df.columns:
        df["summary"] = ""

    summarizer = load_summarizer()

    for idx, row in df.iterrows():
        if not row["summary"]:
            try:
                summary = summarizer(row["full_text"])[0]["summary_text"]
                df.at[idx, "summary"] = summary
            except:
                df.at[idx, "summary"] = ""

    mask = df["summary"].isin(["", None])
    df.loc[mask, "summary"] = df.loc[mask, "full_text"].str.slice(0, 400)

    return df


def get_embeddings(texts):
    return embedder.encode(texts, convert_to_numpy=True)


def final_summary(df, embeddings):
    sim_matrix = cosine_similarity(embeddings)
    centrality = sim_matrix.mean(axis=1)
    df["centrality"] = centrality

    top_idx = np.argsort(-centrality)[:5]
    combined = " ".join(df.iloc[top_idx]["summary"].tolist())

    return summarize(combined)


def compute_score(df, embeddings):
    df["source_name"] = df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

    distinct = df["source_name"].nunique()
    source_score = min(40, distinct * 5)

    dist = df["source_name"].value_counts(normalize=True)
    diversity_score = 30 * (1 - dist.max())

    sim = cosine_similarity(embeddings)
    tri = np.triu_indices_from(sim, k=1)
    avg_sim = sim[tri].mean() if len(tri[0]) > 0 else 0
    agreement_score = avg_sim * 30

    final = min(source_score + diversity_score + agreement_score, 100)
    return {
        "Source Count": round(source_score, 2),
        "Diversity": round(diversity_score, 2),
        "Agreement": round(agreement_score, 2),
        "Final Score (0–100)": round(final, 2)
    }


# ------------------- UI -------------------

st.subheader("🔍 Choose a Topic")

topics = [
    "AI regulation",
    "OpenAI news",
    "India Budget 2025",
    "Ukraine war",
    "Crypto market crash",
    "Nvidia AI chips",
    "ISRO missions",
    "Global inflation",
    "Climate change",
    "US elections",
    "SpaceX launch",
]

selected_topic = st.selectbox("Select a topic (optional):", [""] + topics)
typed_topic = st.text_input("Or type your own topic (optional):")

if typed_topic.strip():
    final_topic = typed_topic.strip()
elif selected_topic.strip():
    final_topic = selected_topic.strip()
else:
    final_topic = ""

st.write("➡️ Final Topic:", final_topic if final_topic else "❗ Please choose or type a topic")

if st.button("🚀 Analyze News"):
    st.info("Fetching and analyzing news...")
    df = fetch_news(final_topic)
    if df.empty:
        st.error("No articles found. Try a different topic.")
    else:
        df = preprocess(df)
        df = add_summaries(df)

        embeddings = get_embeddings(df["summary"].tolist())
        final_sum = final_summary(df, embeddings)
        scores = compute_score(df, embeddings)

        # ------------ OUTPUT ---------------

        st.success("Analysis Complete!")

        st.subheader("📌 Final Consolidated Summary")
        st.write(final_sum)

        st.subheader("📊 Authenticity Score")
        st.write(scores)

        st.subheader("🗞 Timeline of Articles")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(df["publishedAt"], range(len(df)), marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Article Index")
        ax.set_title("Timeline of News Articles")
        st.pyplot(fig)

        st.subheader("📄 News Articles Used")
        st.dataframe(df[["publishedAt", "source_name", "title"]])
