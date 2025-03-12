import re
import random
import string

def remove_identifiers(text):
    # Remove names (simplified version, consider using NLP for better accuracy)
    text = re.sub(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", "Candidate", text)
    
    # Remove phone numbers
    text = re.sub(r"\b\d{10}\b", "##########", text)
    
    # Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "candidate@example.com", text)
    
    # Remove physical addresses (simplified, can be improved)
    text = re.sub(r"\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd)\b", "Candidate Address", text)
    
    return text

def generalize_data(text):
    """
    Generalizes specific companies and job titles in the input text to broader categories
    to anonymize and abstract the data for improved privacy and fairness.
    
    Args:
        text (str): The input text that needs to be generalized.
    
    Returns:
        str: The generalized text.
    """
    # List of companies to generalize
    companies = [
        "Google", "Amazon", "Meta", "Microsoft", "Apple", 
        "Facebook", "Tesla", "Netflix", "IBM", "Oracle", 
        "Twitter", "Adobe", "Intel", "Nvidia", "Salesforce"
    ]
    
    # Replace specific companies with general titles
    for company in companies:
        text = text.replace(company, "Tech Company")
    
    # List of job titles to generalize
    job_titles = [
        "Software Engineer", "Data Scientist", "Machine Learning Engineer", 
        "Project Manager", "Data Analyst", "Business Analyst", 
        "System Architect", "Web Developer", "DevOps Engineer", 
        "Cloud Engineer", "UX/UI Designer", "Full Stack Developer", 
        "QA Engineer", "Product Manager", "Network Engineer", 
        "Database Administrator", "Security Engineer", "Automation Tester", 
        "Mobile Developer", "AI Engineer", "Software Developer"
    ]
    
    # Replace specific job titles with broader roles
    for title in job_titles:
        text = text.replace(title, "Technical Role")
    
    return text


def tokenize_sensitive_words(text):
    """
    Tokenizes sensitive words in the text by replacing them with random tokens.
    
    Args:
        text (str): The input text that needs to be tokenized.
    
    Returns:
        str: The text with sensitive words replaced by random tokens.
    """
    # List of sensitive words to replace
    sensitive_words = [
        "AI", "ML", "Python", "TensorFlow", "NLP", 
        "Google", "Amazon", "Meta", "Microsoft", "Apple", 
        "Facebook", "Tesla", "Netflix", "IBM", "Oracle", 
        "Twitter", "Adobe", "Intel", "Nvidia", "Salesforce",
        "Software Engineer", "Data Scientist", "Machine Learning Engineer", 
        "Project Manager", "Data Analyst", "Business Analyst", 
        "System Architect", "Web Developer", "DevOps Engineer", 
        "Cloud Engineer", "UX/UI Designer", "Full Stack Developer", 
        "QA Engineer", "Product Manager", "Network Engineer", 
        "Database Administrator", "Security Engineer", "Automation Tester", 
        "Mobile Developer", "AI Engineer", "Software Developer"
    ]
    
    # Replace sensitive words with random tokens
    for word in sensitive_words:
        token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        text = text.replace(word, token)
    
    return text

def anonymize_resume(resume_text):
    resume_text = remove_identifiers(resume_text)
    resume_text = generalize_data(resume_text)
    final_resume_text = tokenize_sensitive_words(resume_text)
    return final_resume_text
