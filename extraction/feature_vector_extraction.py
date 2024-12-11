# extraction/feature_vector_extraction.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def extract_tfidf_features_from_resumes(resume_list):
    """
    Extracts TF-IDF feature vectors from a list of resume texts.

    Args:
    resume_list (list): A list of resume text strings.

    Returns:
    pd.DataFrame: DataFrame containing the TF-IDF features for each resume.
    """
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the model on the text data and transform it into a sparse matrix
    tfidf_matrix = vectorizer.fit_transform(resume_list)

    # Get the feature names (i.e., the words/tokens) from the fitted model
    feature_names = vectorizer.get_feature_names_out()

    # Convert the sparse matrix into a dense DataFrame with feature names as columns
    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


def extract_embeddings_from_resumes(resumes, model_name='all-MiniLM-L6-v2'):
    """
    Extracts sentence-level embeddings from a list or Series of resumes using Sentence-BERT.

    Args:
    resumes (list or pd.Series): A list or pandas Series containing resume texts.
    model_name (str): The name of the Sentence-BERT model to use (default: 'all-MiniLM-L6-v2').

    Returns:
    pd.DataFrame: DataFrame containing the embeddings for the resumes.
    """
    # Load sentence-BERT model
    model = SentenceTransformer(model_name)
    
    # Encode resumes into embeddings
    embeddings = model.encode(resumes, show_progress_bar=True)
    
    # Create a new DataFrame from the embeddings
    embedding_df = pd.DataFrame(embeddings, index=resumes.index if isinstance(resumes, pd.Series) else None)
    
    return embedding_df
