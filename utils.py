import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

def extract_keywords(text, top_n=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words='english', max_features=top_n)
    vec.fit([text])
    return vec.get_feature_names_out()
