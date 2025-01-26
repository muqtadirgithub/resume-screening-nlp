import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
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



# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

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

            resume_embeddings = sbert_model.encode(processed_resumes)
            jd_embedding = sbert_model.encode([job_description])[0]

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
