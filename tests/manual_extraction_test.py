from matching.extraction_utils import extract_skills_edu_exp

resume_text = """
John Doe is a software engineer skilled in Python, TensorFlow, and cloud platforms like AWS.
He has a Bachelor's degree in Computer Science from Stanford University.
Worked as a data scientist at XYZ Corp and interned with IBM Watson AI Labs.
"""

results = extract_skills_edu_exp(resume_text)

print("Extracted Info:")
for line in results:
    print("-", line)
