from classification.classify_resumes import classify_resumes

resumes = [
    "Skilled in Python, TensorFlow, and NLP. Worked on chatbots and transformers.",
    "Experienced Java developer with strong backend experience.",
    "Managed HR teams, recruited engineers, and handled payroll systems."
]

labels = [
    "AI Engineer",
    "Backend Developer",
    "HR Manager"
]

new_resume = "Experience with Python, NLP, chatbots, and AI tools like Hugging Face."

prediction = classify_resumes(resumes, labels, new_resume)
print(f"Predicted role: {prediction}")
