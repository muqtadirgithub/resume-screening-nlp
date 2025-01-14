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
