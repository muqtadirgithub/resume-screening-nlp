# 📝 Resume Screening NLP

A Streamlit-based application that uses NLP and machine learning to classify and rank resumes based on a job description.

## 🚀 Features

- Upload and parse multiple resume PDFs
- Extract skills, education, and experience
- Match resumes semantically to job description using SBERT
- Predict job role category using a trained classification model
- Interactive UI with real-time results and rankings

## 📦 Requirements

- Python 3.8+
- `streamlit`, `sklearn`, `nltk`, `PyPDF2`, `sentence-transformers`, `pandas`

## 💻 Usage

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
