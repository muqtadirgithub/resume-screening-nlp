from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compare_resume_to_jd(resume_text, job_description):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_vec = model.encode([resume_text])[0].reshape(1, -1)
    jd_vec = model.encode([job_description])[0].reshape(1, -1)
    score = cosine_similarity(resume_vec, jd_vec)[0][0]
    return round(score, 4)
