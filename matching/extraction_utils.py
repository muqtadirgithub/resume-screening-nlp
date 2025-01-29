def extract_skills_edu_exp(resume_text):
    """Function to extract skill, education, and work experience-related lines from resume text"""
    import re

    skills_keywords = {
        "skills", "skill", "technical skills", "soft skills", "tools", "technologies", "technology",
        "frameworks", "libraries", "platforms", "languages", "certifications", "methodologies",
        "programming", "databases", "cloud", "devops", "analytics", "testing tools", "networking"
    }

    education_keywords = {
        "bachelor", "masters", "phd", "degree", "university", "college",
        "education", "diploma", "b.tech", "mba", "mca", "engineering",
        "computer science", "information technology", "cs", "it", "software", "technology",
        "business", "finance", "management", "law", "medical", "arts", "science", "commerce"
    }

    work_keywords = {
        "experience", "worked", "employed", "company", "organization", "intern", "internship"
    }

    skills_pattern = re.compile(r'\b(' + '|'.join(skills_keywords) + r')\b', re.IGNORECASE)
    edu_pattern = re.compile(r'\b(' + '|'.join(education_keywords) + r')\b', re.IGNORECASE)
    work_pattern = re.compile(r'\b(' + '|'.join(work_keywords) + r')\b', re.IGNORECASE)

    skills, education, experience = [], [], []
    lines = resume_text.split('\n')

    for line in lines:
        sentence = line.strip()
        lower = sentence.lower()

        if skills_pattern.search(lower):
            skills.append(sentence)
        if edu_pattern.search(lower):
            education.append(sentence)
        if work_pattern.search(lower):
            experience.append(sentence)

    return skills + education + experience

