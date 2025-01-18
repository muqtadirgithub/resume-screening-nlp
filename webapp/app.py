import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
