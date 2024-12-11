from extraction.feature_vector_extraction import extract_embeddings_from_resumes
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes_by_semantic_similarity(job_description, resumes):
    """
    Ranks resumes based on their semantic similarity to a given job description.

    Args:
        job_description (str): The job description text.
        resumes (list): A list of resume texts.

    Returns:
        list: List of tuples (index, similarity_score) sorted by similarity_score in descending order.
    """

    # Generate embedding for job description using the function from extract_embeddings_from_resumes.py
    job_des_embedding = extract_embeddings_from_resumes([job_description]).values[0].reshape(1, -1)

    # Generate embeddings for all resumes using the same function
    resume_embeddings = extract_embeddings_from_resumes(resumes).values  # Now a proper 2D array

    # Calculate cosine similarity for each resume
    similarities = cosine_similarity(job_des_embedding, resume_embeddings)[0]

    # Rank resumes by their similarity score
    ranked_resumes = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    return ranked_resumes
