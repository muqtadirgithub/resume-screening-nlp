import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from extraction.feature_vector_extraction import extract_embeddings_from_resumes

from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load trained model and label encoder
model = joblib.load("resume_classification_model.pkl")
label_encoder = joblib.load("resume_label_encoder.pkl")

# Streamlit page config
st.set_page_config(page_title="Resume Screening App", layout="centered")
st.title("📝 Resume Screening NLP")
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This app lets you upload multiple resumes and match them against a job description using AI.

- Uses **SBERT** for semantic similarity
- Uses a trained classifier for job role prediction
- Extracts skills, education, and experience
""")
st.markdown("Upload resumes and classify them based on job categories using NLP and ML.")
from PyPDF2 import PdfReader

# Initialize session state
if "text_blocks" not in st.session_state:
    st.session_state.text_blocks = []

if "file_names" not in st.session_state:
    st.session_state.file_names = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Upload resumes
st.markdown("## 📄 Upload Resumes (PDF)")
uploaded_files = st.file_uploader(
    "Upload Multiple Resume Files",
    type="pdf",
    accept_multiple_files=True
)

# Extract text from resumes
if uploaded_files:
    st.markdown("### ✅ Extracted Resume Text")
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if text.strip():
                st.session_state.text_blocks.append(text.strip())
                st.session_state.file_names.append(uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"Extracted text from: {uploaded_file.name}")





# Job description input
st.markdown("## 🧾 Paste Job Description")
job_description = st.text_area("Enter the job description here:")

# Similarity calculation
if st.button("Match Resumes"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not st.session_state.text_blocks:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Ranking resumes..."):

           
            processed_resumes = [preprocess_resume(resume) for resume in st.session_state.text_blocks]

            resume_embeddings_df = extract_embeddings_from_resumes(processed_resumes)
            jd_embedding_df = extract_embeddings_from_resumes([job_description])

            resume_embeddings = resume_embeddings_df.values
            jd_embedding = jd_embedding_df.values[0].reshape(1, -1)


            similarities = cosine_similarity([jd_embedding], resume_embeddings)[0]

            # Rank and show
            ranked = sorted(zip(st.session_state.file_names, similarities), key=lambda x: -x[1])
            st.markdown("## 📊 Resume Match Results")
            for name, score in ranked:
                st.write(f"**{name}** — Similarity Score: `{score:.4f}`")
            # Predict categories for each resume
            predicted_labels = model.predict(resume_embeddings)
            predicted_categories = label_encoder.inverse_transform(predicted_labels)

            # Create DataFrame of results
            result_df = pd.DataFrame({
                "Resume File": st.session_state.file_names,
                "Similarity Score": similarities,
                "Predicted Category": predicted_categories
            })

            # Add ranking
            result_df["Rank"] = result_df["Similarity Score"].rank(ascending=False, method='first').astype(int)
            result_df = result_df.sort_values(by="Rank")

            st.markdown("## 🧠 Predicted Categories and Resume Rankings")
            st.dataframe(result_df[["Resume File", "Similarity Score", "Predicted Category", "Rank"]])
import streamlit as st
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load trained model and label encoder
model = joblib.load("resume_classification_model.pkl")
label_encoder = joblib.load("resume_label_encoder.pkl")

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Preprocessing Functions ----------

def preprocess_resume(resume_text):
    """
    Basic NLP cleaning: lowercase, remove punctuation, tokenize, remove stopwords, lemmatize.
    """
    text = resume_text.lower()
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def extract_skills_edu_exp(resume_text):
    """Extract skill, education, and experience-related lines from resume text"""
    skills_keywords = {
        "skills", "skill", "technical skills", "soft skills", "tools", "technologies", "technology",
        "frameworks", "libraries", "platforms", "languages", "certifications", "methodologies",
        "programming", "databases", "cloud", "devops", "analytics", "testing tools", "networking"
    }
    education_keywords = {
        "bachelor", "bachelors", "master", "masters", "phd", "degree", "degrees", "university", "college",
        "graduate", "postgraduate", "undergraduate", "school", "education", "certification", "certifications", "diploma",
        "b.tech", "be", "bsc", "bca", "bba", "ba", "mba", "mca", "m.tech", "me", "msc", "ms", "pgdm", "pg", "ug", "llb", "llm",
        "data science", "machine learning", "artificial intelligence", "computer science", "cs",
        "information technology", "it", "software engineering", "cybersecurity", "network security",
        "devops", "cloud computing", "aws", "azure", "gcp", "database", "big data", "hadoop", "etl",
        "python", "java", "dotnet", ".net", "blockchain", "web development", "web designing", "ui ux",
        "software testing", "automation testing", "qa", "full stack", "frontend", "backend",
        "business administration", "business analytics", "operations management", "operations",
        "human resources", "hr", "marketing", "finance", "accounting", "commerce", "economics",
        "entrepreneurship", "strategy", "organizational behavior", "project management", "pmp",
        "mechanical engineering", "civil engineering", "electrical engineering", "electronics",
        "electronics and communication", "ece", "instrumentation", "engineering",
        "law", "legal studies", "advocate", "llb", "llm", "jurisprudence", "judicial", "legal",
        "liberal arts", "fine arts", "performing arts", "visual arts", "design", "health sciences",
        "healthcare", "nursing", "medicine", "fitness", "physical education", "nutrition",
        "communication", "leadership", "pmo", "analyst", "consultant", "trainer", "coach"
    }
    work_keywords = {
        "experience", "exprience", "worked", "employed", "company", "organization", "intern", "internship"
    }

    skills_pattern = re.compile(r'\b(' + '|'.join(skills_keywords) + r')\b', re.IGNORECASE)
    edu_pattern = re.compile(r'\b(' + '|'.join(education_keywords) + r')\b', re.IGNORECASE)
    work_pattern = re.compile(r'\b(' + '|'.join(work_keywords) + r')\b', re.IGNORECASE)

    skills, education, experience = [], [], []
    lines = resume_text.split('\n')

    for line in lines:
        sentence = line.strip()
        lower = sentence.lower()
        if skills_pattern.search(lower):
            skills.append(sentence)
        if edu_pattern.search(lower):
            education.append(sentence)
        if work_pattern.search(lower):
            experience.append(sentence)

    return skills + education + experience

# ---------- Streamlit UI Setup ----------

st.set_page_config(page_title="Resume Screening App", layout="centered")
st.title("📝 Resume Screening NLP")
st.markdown("Upload resumes and classify them based on job categories using NLP and ML.")

# ---------- File Upload and Text Extraction ----------

if "text_blocks" not in st.session_state:
    st.session_state.text_blocks = []
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

st.markdown("## 📄 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload Multiple Resume Files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.markdown("### ✅ Extracted Resume Text")

    any_valid = False  # Flag to track if at least one resume is readable

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            reader = PdfReader(uploaded_file)
            text = ""

            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    st.warning(f"⚠️ Error reading {uploaded_file.name}: {e}")

            cleaned_text = text.strip()
            if cleaned_text:
                any_valid = True
                st.session_state.text_blocks.append(cleaned_text)
                st.session_state.file_names.append(uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"✅ Extracted text from: {uploaded_file.name}")
            else:
                st.warning(f"⚠️ No extractable text found in: {uploaded_file.name}")

    if not any_valid:
        st.error("❌ No valid resume content found. Please check your files.")

# ---------- Job Description Input & Matching ----------

st.markdown("## 🧾 Paste Job Description")
job_description = st.text_area("Enter the job description here:")

if st.button("Match Resumes"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not st.session_state.text_blocks:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Ranking resumes..."):

            processed_resumes = [preprocess_resume(r) for r in st.session_state.text_blocks]
            resume_embeddings = sbert_model.encode(processed_resumes)
            jd_embedding = sbert_model.encode([job_description])[0]

            similarities = cosine_similarity([jd_embedding], resume_embeddings)[0]

            ranked = sorted(zip(st.session_state.file_names, similarities), key=lambda x: -x[1])
            st.markdown("## 📊 Resume Match Results")
            for name, score in ranked:
                st.write(f"**{name}** — Similarity Score: `{score:.4f}`")

            predicted_labels = model.predict(resume_embeddings)
            predicted_categories = label_encoder.inverse_transform(predicted_labels)

            result_df = pd.DataFrame({
                "Resume File": st.session_state.file_names,
                "Similarity Score": similarities,
                "Predicted Category": predicted_categories
            })

            result_df["Rank"] = result_df["Similarity Score"].rank(ascending=False, method='first').astype(int)
            result_df = result_df.sort_values(by="Rank")

            st.markdown("## 🧠 Predicted Categories and Resume Rankings")
            st.dataframe(result_df[["Resume File", "Similarity Score", "Predicted Category", "Rank"]])
 
            st.markdown("## 🧾 Extracted Resume Details")

            for i, resume_text in enumerate(st.session_state.text_blocks): 
                file_name = st.session_state.file_names[i]
                extracted_lines = extract_skills_edu_exp(resume_text)
                st.markdown(f"### 📄 {file_name}")
                if extracted_lines:
                    for line in extracted_lines:
                        st.write(f"- {line}")
                else:
                    st.warning("No relevant details extracted.")

