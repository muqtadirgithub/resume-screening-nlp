import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

from preprocessing.preprocessing import preprocess_resume 
from extraction.extract_skills_edu_exp import extract_skills_edu_exp
from extraction.extract_custom_skills_spacy import extract_custom_skills_using_phrasematcher
from extraction.feature_vector_extraction import extract_tfidf_features_from_resumes,extract_embeddings_from_resumes  
from classification.classify_resumes import classify_resumes
from matching.rank_resumes_by_similarity import rank_resumes_by_semantic_similarity
from anonymization.anonymization import anonymize_resume

# Function to fix encoding issues in text
def fix_encoding(text):
    return text.encode('latin1').decode('utf-8', errors='ignore')  # or use errors='replace'

if __name__ == "__main__":
    # Step 1: Load the resume data from a CSV file
    csv_file = r'data/UpdatedResumeDataSet.csv'
    column_name = 'Resume'
    resume_df = pd.read_csv(csv_file, encoding='utf-8')

    # Step 2: Fix any encoding issues in the Resume column
    resume_df['Resume'] = resume_df['Resume'].apply(fix_encoding)

    resume_df['Resume'] = resume_df['Resume'].apply(anonymize_resume)

    # Step 3: Preprocess the resume text (e.g., lowercase, remove stopwords, clean punctuation)
    resume_df["Resume_processed"] = resume_df["Resume"].astype(str).apply(preprocess_resume)

    # Step 4: Extract structured information - Skills, Education, and Experience - from the original resume text
    resume_df[['Skills', 'Education', 'Experience']] = resume_df['Resume'].apply(extract_skills_edu_exp)
    resume_df["skills_spacy"] = resume_df['Resume'].apply(extract_custom_skills_using_phrasematcher)
    resume_df.to_csv("output.csv",index=False)

    # Step 5: Apply TF-IDF vectorization on the preprocessed resume text
    resumes = resume_df["Resume_processed"].fillna('')  # Ensure there are no NaN values
    tfidf_vectors_df = extract_tfidf_features_from_resumes(resumes)

    # Display the TF-IDF feature matrix (each row is a resume, each column is a term)
    print(tfidf_vectors_df)

    # Step 6: Generate semantic embeddings for the resumes using Sentence-BERT (SBERT)
    sbert_embeddings_df = extract_embeddings_from_resumes(resumes)

    # Display the SBERT embeddings DataFrame (each row is a vector representation of a resume)
    print(sbert_embeddings_df)


    # Train the model and print the classification report
    classify_resumes(resume_df["Resume"], resume_df["Category"])
    
    print("Resume_processed-------------------")
    resumes = resume_df["Resume_processed"][0:20]
    job_description = """
    Skills:

    Proven experience of 6 + years as an Data Scientist

    Solid experience in data science ML modelling (Classification, Regressions, time series, clustering, pattern recognition) optimization, reinforcement learning, deep learning algorithms, cloud computing architecture and big data management and reporting, NLP and Generative AI
    Knowledge of Python, SQL , Azure ML
    Familiarity with business intelligence tools/ data Visualization tools - Power BI
    Demonstrates an understanding of the IT environment, including enterprise applications like HRMS, ERPs, CRM, etc.

    Educational Qualification:

    Graduate/Post Graduate degree Computer Science, Statistics, Data Science or a

    related field
    """
    ranked_results = rank_resumes_by_semantic_similarity(job_description, resumes)

    # Display ranked resumes
    print(f"{'Resume':<8} | {'Index':<5} | {'Score':<7} | {'Rank':<4}")
    print("-" * 36)
    for rank, (idx, score) in enumerate(ranked_results, start=1):
        print(f"{'Resume':<8} | {idx:<5} | {score:<7.4f} | {rank:<4}")

    print("Resume--------------------------")
    resumes = resume_df["Resume"][0:20]
    ranked_results = rank_resumes_by_semantic_similarity(job_description, resumes)

    # Display ranked resumes
    print(f"{'Resume':<8} | {'Index':<5} | {'Score':<7} | {'Rank':<4}")
    print("-" * 36)
    for rank, (idx, score) in enumerate(ranked_results, start=1):
        print(f"{'Resume':<8} | {idx:<5} | {score:<7.4f} | {rank:<4}")

    resume_df['skills_edu_exp_resume'] = resume_df[['Skills', 'Education', 'Experience']].astype(str).agg(' '.join, axis=1)

    print("skills_edu_exp_resume ----------------------------")
    resumes = resume_df["skills_edu_exp_resume"][0:20]
    ranked_results = rank_resumes_by_semantic_similarity(job_description, resumes)

    # Display ranked resumes
    print(f"{'Resume':<8} | {'Index':<5} | {'Score':<7} | {'Rank':<4}")
    print("-" * 36)
    for rank, (idx, score) in enumerate(ranked_results, start=1):
        print(f"{'Resume':<8} | {idx:<5} | {score:<7.4f} | {rank:<4}")