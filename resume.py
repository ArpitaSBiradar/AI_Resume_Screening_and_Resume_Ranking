import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import pdfkit
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
import os
import openai

# Load OpenAI API Key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------- Styling --------------------------
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    /* Set Background */
    .stApp {
        background-color: #FFFFFF;
        color: black;
    }

    /* Main Title */
    .main-title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
        padding: 5px;  /* Adjusted spacing */
        margin-top: -30px;  /* Moves the header up */
    }

    /* Sidebar */
    .stSidebar {
        background-color: #D8BFD8;
        padding: 10px;
        border-radius: 10px;
        font-size: 18px;  /* Increased sidebar text size */
    }

    /* Resume Ranking Card */
    .resume-card {
        padding: 15px;
        background: #F8F9FA;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }

    /* Animated Score */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    .score-text {
        font-size: 22px;
        font-weight: bold;
        color: #27AE60;
        animation: fadeIn 2s;
    }

    /* Best Match */
    .best-match {
        font-size: 24px;
        font-weight: bold;
        color: black;  /* Changed to black */
        text-align: center;
        padding: 15px;
        background: #FFD700; /* Gold background for visibility */
        border-radius: 10px;
        margin-top: 20px;
    }

    /* Download Button */
    .download-button {
        display: block;
        text-align: center;
        padding: 12px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #FFd470;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">⚡ Quick Actions</h2>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("📤 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    job_description = st.text_area("📝 Enter Job Description")

    # Dark Mode Toggle
    dark_mode = st.checkbox("🌙 Dark Mode")
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp { background-color: #333; color: white; }
            .resume-card { background: #444; color: white; }
            .best-match { background: #666; color: white; }  /* Best match in dark mode */
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("📌 **Guidelines:**")
    st.markdown("[🔗 How to Write a Resume?](https://www.resume.com/)")

    # About Section
    st.markdown("---")
    st.markdown("👨‍💻 *Developed by:* Arpita Biradar")
    st.markdown("[🔗 Know More About Us](https://www.linkedin.com/in/arpita-s-biradar-344811298/)")



# -------------------------- Main Content --------------------------
st.markdown('<h1 class="main-title">📄 AI Resume Screening & Ranking System</h1>', unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to Rank Resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_description_vector], resume_vectors).flatten()

# Generate Word Cloud (Reduced Size)
def generate_word_cloud(text):
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(text)  # Reduced size
    plt.figure(figsize=(8, 4))  # Adjusted figure size
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Processing Resumes
if uploaded_files and job_description:
    st.markdown('<h2 class="sub-title">🏆 Ranking Resumes</h2>', unsafe_allow_html=True)

    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    for i, row in results.iterrows():
        st.markdown(
            f"""
            <div class="resume-card">
                <h3>📑 {row['Resume']}</h3>
                <p class="score-text">Matching Score: <b>{row['Score']*100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(row["Score"])

    # Show Best Match
    top_resume = results.iloc[0]
    st.markdown(
        f"<div class='best-match'>🎯 Best Match: <b>{top_resume['Resume']}</b> with <b>{top_resume['Score']*100:.2f}% match!</b></div>",
        unsafe_allow_html=True,
    )

    # Generate Word Cloud
    st.markdown('<h2 class="sub-title">☁️ Resume Word Cloud</h2>', unsafe_allow_html=True)
    generate_word_cloud(" ".join(resumes))

    # CSV Download Button
    csv = results.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    st.markdown(
        f'<a href="data:file/csv;base64,{b64}" download="resume_ranking.csv" class="download-button">📥 Download Ranking Report</a>',
        unsafe_allow_html=True,
    )
