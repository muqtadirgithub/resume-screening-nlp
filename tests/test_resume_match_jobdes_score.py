from matching.match_inspector import compare_resume_to_jd

def test_score_resume_against_job():
    resume = "Experienced software engineer with Python, ML, and NLP background. Built chatbot systems using transformers."
    jd = "Looking for a machine learning engineer with NLP experience, preferably in Python and chatbot systems."
    
    score = compare_resume_to_jd(resume, jd)
    print(f"🧪 Similarity Score: {score}")
    assert score > 0.5, "Score too low — match logic may be off"

if __name__ == "__main__":
    test_score_resume_against_job()
