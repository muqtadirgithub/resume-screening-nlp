
from matching.rank_resumes_by_similarity import rank_resumes_by_semantic_similarity

job_description = "Looking for a software engineer with Python, ML, and NLP experience."
resumes = [
    "Experienced Python developer with strong background in machine learning and text analytics.",
    "Front-end engineer with React and TypeScript skills.",
    "Recent CS graduate with internship in NLP research."
]

if __name__ == "__main__":
    rankings = rank_resumes_by_semantic_similarity(job_description, resumes)
    print("Ranked Resumes (index, score):")
    for index, score in rankings:
        print(f"{index}: {score:.4f}")
