# Resume Screening and Ranking System

## 📌 Project Overview

This project is a comprehensive **Resume Screening and Ranking System** that automates the process of parsing resumes, extracting relevant information (such as skills, education, and experience), and ranking candidates based on their suitability for a given job description. 
The system leverages **Natural Language Processing (NLP)**, **Machine Learning**, and a simple **web interface** for an end-to-end solution.

---

## 🧩 Features

- ✅ **Resume Data Ingestion and Preprocessing**
- ✅ **Named Entity Recognition (NER)** for skills, education, and experience
- ✅ **Feature extraction** using TF-IDF and Sentence-BERT
- ✅ **Custom skill extraction** with spaCy Matcher and NER
- ✅ **Semantic matching** using cosine similarity
- ✅ **Candidate ranking** with skill overlap metric
- ✅ **Optional classification** by job categories
- ✅ **Optional fairness and bias mitigation**
- ✅ **Web-based interface** using Streamlit or Flask

## 📂 Project Structure
```bash
resume-screening-nlp/
│
├── anonymization/ # Data anonymization and masking
├── classification/ # Model training and classification logic
├── extraction/ # Embedding and skills/experience extraction
├── matching/ # Resume-job description similarity ranking
├── preprocessing/ # Text cleaning and preprocessing
├── model/ # Trained model and label encoders (generated)
├── tests/ # Unit and integration tests
├── webapp/ # Streamlit or Flask web interface
├── main.py # Entry point to run the pipeline
├── requirements.txt # Python package dependencies
└── README.md # Project documentation
```

## 🚀 Setup Instructions

Python Version : 3.10.10 (Used lightning.ai)

Clone the repository:

```bash
git clone https://github.com/muqtadirgithub/resume-screening-nlp.git
cd resume-screening-nlp
```

## Running the Application
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

```bash
python main.py
```

## 🧪 Running Tests
```bash
PYTHONPATH=. pytest
```
