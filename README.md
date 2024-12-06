# Resume Screening NLP

This project automates the classification and ranking of resumes using machine learning and NLP techniques.

## Overview

We aim to streamline the resume screening process by:
- Classifying resumes into predefined job categories
- Matching resumes against a specific job description
- Ranking resumes based on semantic similarity

## Modules

- **classification/**: Train and use machine learning models to classify resumes.
- **matching/**: Compare resumes to job descriptions and rank based on relevance.
- **webapp/**: A Flask frontend to interact with the system.
- **tests/**: Unit and integration tests for all modules.

## Getting Started

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the web app:
    ```bash
    python webapp/app.py
    ```
