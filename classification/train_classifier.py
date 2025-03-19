import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from extraction.feature_vector_extraction import extract_embeddings_from_resumes
from preprocessing.clean_text import preprocess_text  # assumed from your structure

# Load dataset
df = pd.read_csv("data/UpdatedResumeDataSet.csv").dropna()
texts = df["Resume"].tolist()
labels = df["Category"].tolist()

# Preprocess
processed = [preprocess_text(t) for t in texts]

# Embed
X = extract_embeddings_from_resumes(processed)
le = LabelEncoder()
y = le.fit_transform(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model + label encoder
joblib.dump(model, "webapp/resume_classification_model.pkl")
with open("webapp/resume_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
