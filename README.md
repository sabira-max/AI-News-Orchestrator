# 📰 AI News Orchestrator

AI News Orchestrator is a Streamlit-based application that fetches latest news from multiple sources, summarizes them using AI, generates embeddings for semantic similarity, and computes an authenticity score.

This helps users get:
- ✔ A clean consolidated summary
- ✔ Multi-source validation
- ✔ Timeline of events
- ✔ Source diversity score
- ✔ Agreement score between sources

---

## 🚀 Features

### 🔍 Topic Search
Search news by:
- Dropdown topic list
- Or type your own topic manually

### 🤖 AI Summarization
Uses `t5-small` transformer model to generate accurate short summaries.

### 🧠 Embedding & Similarity
Uses Sentence-Transformer (`all-MiniLM-L6-v2`) to compute:
- Source agreement  
- News similarity  
- Centrality-based final summary  

### 📊 Authenticity Score (0–100)
Based on:
- Source count  
- Diversity  
- Agreement across sources  

### 🕔 Timeline Visualization
Scatter plot showing publishing times of articles.

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Transformers
- Sentence Transformers
- Sklearn
- Matplotlib
- NewsAPI

---

## 📦 Installation

```bash
git clone https://github.com/sabira-max/AI-News-Orchestrator.git
cd AI-News-Orchestrator
pip install -r requirements.txt
streamlit run AI_News.py



