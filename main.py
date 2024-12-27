from classification.classify_resumes import classify_resumes

if __name__ == "__main__":
    resumes = [
        "Experienced Python developer with machine learning and NLP background.",
        "Front-end developer skilled in React and TypeScript.",
        "Data analyst with experience in SQL, Excel, and Tableau.",
        "AI researcher with focus on deep learning and transformer models."
    ]

    labels = [
        "Machine Learning Engineer",
        "Frontend Developer",
        "Data Analyst",
        "AI Researcher"
    ]

    new_resume = "Skilled in Python, NLP, and deploying ML models."

    prediction = classify_resumes(resumes, labels, new_resume)
    print(f"Predicted Role: {prediction}")
