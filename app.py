import streamlit as st
import joblib
import pdfplumber
from docx import Document
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tempfile
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Classification System",
    layout="centered",
    page_icon="📄"
)

# ---------------- LOAD NLTK ----------------
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = load_stopwords()

# ---------------- LOAD MODEL & VECTORIZER ----------------
@st.cache_resource
def load_models():
    svm_model = joblib.load("resume_classifier_svm.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return svm_model, tfidf_vectorizer

try:
    svm_model, tfidf_vectorizer = load_models()
except Exception:
    st.error("❌ Model files not found. Please check repository files.")
    st.stop()

# ---------------- LABEL NORMALIZATION ----------------
LABEL_ALIAS = {
    "react js developer": "react",
    "sql developer": "sql",
    "peoplesoft admin": "peoplesoft",
    "workday": "workday",
    "internship": "internship"
}

# ---------------- SKILL MAP ----------------
SKILL_MAP = {
    "react": [
        "react", "javascript", "html", "css", "redux",
        "hooks", "bootstrap", "material ui",
        "rest api", "git", "webpack", "typescript"
    ],
    "sql": [
        "sql", "mysql", "postgresql", "oracle",
        "joins", "index", "procedure", "etl",
        "view", "trigger", "performance", "tuning"
    ],
    "peoplesoft": [
        "peoplesoft", "peopletools", "hrms", "hcm",
        "application engine", "sqr", "ps query",
        "bi publisher", "event mapping"
    ],
    "workday": [
        "workday", "hcm", "integration", "studio", "eib", "reporting"
    ],
    "internship": [
        "python", "java", "sql", "html",
        "css", "javascript", "data analysis"
    ]
}

# ---------------- HELPER FUNCTIONS ----------------
def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def read_docx(path):
    doc = Document(path)
    return " ".join(p.text for p in doc.paragraphs)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\b\d{10,12}\b', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(w for w in text.split() if w not in stop_words and len(w) > 2)


# ---------------- UI ----------------
st.title("📄 Resume Classification System")
st.write(
    "Upload a resume (**PDF or DOCX**) to predict the job category "
    "and extract important skills."
)

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Extract text
    if uploaded_file.name.lower().endswith(".pdf"):
        raw_text = read_pdf(temp_path)
    else:
        raw_text = read_docx(temp_path)

    os.remove(temp_path)

    if not raw_text.strip():
        st.warning("⚠️ Could not extract text from the resume.")
        st.stop()

    cleaned_text = clean_text(raw_text)

    # -------- CATEGORY PREDICTION --------
    vector = tfidf_vectorizer.transform([cleaned_text])
    prediction = svm_model.predict(vector)[0]

    st.success(f"✅ Predicted Category: **{prediction}**")

    # -------- SKILL EXTRACTION --------
    st.subheader("🛠️ Key Skills Detected")

    role_key = LABEL_ALIAS.get(prediction.lower().strip())
    detected_skills = []

    if role_key and role_key in SKILL_MAP:
        for skill in SKILL_MAP[role_key]:
            if re.search(rf"\b{re.escape(skill)}\b", cleaned_text):
                detected_skills.append(skill)

    if detected_skills:
        st.write(", ".join(f"✔️ {s.title()}" for s in sorted(set(detected_skills))))
    else:
        st.info("No predefined skills detected.")

    # -------- WORD FREQUENCY --------
    st.subheader("📊 Top 10 Most Common Words")
    words = cleaned_text.split()
    freq = Counter(words).most_common(10)

    if freq:
        labels, counts = zip(*freq)
        fig, ax = plt.subplots()
        ax.barh(labels[::-1], counts[::-1])
        ax.set_xlabel("Frequency")
        st.pyplot(fig)

    # -------- WORD CLOUD --------
    st.subheader("☁️ Word Cloud")
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(cleaned_text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # -------- RAW TEXT --------
    with st.expander("📄 View Extracted Resume Text"):
        st.write(raw_text[:3000])

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <center>
    Built by <b>Thousifulla</b> | Resume Classification using NLP & ML
    </center>
    """,
    unsafe_allow_html=True
)
