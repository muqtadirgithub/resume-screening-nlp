import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from extraction.feature_vector_extraction import extract_embeddings_from_resumes  # Import the function

def classify_resumes(resumes, labels, new_resume=None):
    """
    Preprocesses resumes, trains a classifier, and optionally classifies a new resume.
    
    Args:
        resumes (list): List of resume texts.
        labels (list): List of job category labels.
        new_resume (str, optional): A new resume text to classify. Default is None.
        
    Returns:
        str: If `new_resume` is provided, returns the predicted job category for the new resume.
        None: Otherwise, trains and evaluates the model and prints classification report.
    """
    # Encode resumes using extract_embeddings_from_resumes function
    X = extract_embeddings_from_resumes(resumes)
    
    # Encode job labels into numeric values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Logistic Regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Save model and label encoder for later use
    with open('model/resume_classification_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

    with open('model/resume_label_encoder.pkl', 'wb') as file:
        pickle.dump(le, file)
    
    # If a new resume is provided, classify it
    if new_resume:
        # Encode the new resume text
        embedding = extract_embeddings_from_resumes([new_resume])
        
        # Predict the job category
        pred = clf.predict(embedding)
        
        # Convert the numerical prediction back to the original label
        return le.inverse_transform(pred)[0]
    else:
        # No new resume, just return None
        return None


def predict_resume_category(resume_texts):
    """
    Given a list or Series of resume texts, return their predicted categories.
    """
    # Load pre-trained components
    model = joblib.load(r"model/resume_classification_model.pkl")
    tokenizer = joblib.load(r"model/resume_label_encoder.pkl")

    if isinstance(resume_texts, str):
        resume_texts = [resume_texts]

    # Generate embeddings
    embeddings = extract_embeddings_from_resumes(resume_texts)

    # Predict class labels
    predicted_labels = model.predict(embeddings)

    # Decode labels to category names
    predicted_categories = tokenizer.inverse_transform(predicted_labels)

    return predicted_categories