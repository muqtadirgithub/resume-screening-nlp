from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ResumeClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def train(self, texts, labels):
        self.vectorizer.fit_transform(texts)  
        self.model.fit(texts, labels)         

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
