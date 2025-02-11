from webapp.app import extract_skills_edu_exp

def test_extraction():
    resume_text = """
    Jane Doe has a Master's degree in Computer Science.
    She worked as a data scientist and is skilled in Python, TensorFlow, and SQL.
    Also experienced in cloud platforms like AWS.
    """

    extracted = extract_skills_edu_exp(resume_text)

    assert any("computer science" in line.lower() for line in extracted), "Education not detected"
    assert any("python" in line.lower() for line in extracted), "Skill not detected"
    assert any("worked" in line.lower() for line in extracted), "Experience not detected"

    print("✅ All extraction assertions passed.")

if __name__ == "__main__":
    test_extraction()
