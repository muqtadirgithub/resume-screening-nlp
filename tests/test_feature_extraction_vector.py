import unittest
from extraction.feature_vector_extraction import extract_tfidf_features_from_resumes,extract_embeddings_from_resumes 
import numpy as np
import pandas as pd

class TestFeatureExtraction(unittest.TestCase):

    # ---------- Tests for vectorize_resumes ----------
    def test_vectorize_resumes_returns_non_empty_vector(self):
        text = "hello world"
        features = extract_tfidf_features_from_resumes([text])
        self.assertGreater(len(features), 0)

    def test_vectorize_resumes_returns_numpy_array(self):
        features = extract_tfidf_features_from_resumes(["sample text"])
        self.assertIsInstance(features, pd.DataFrame)

    def test_vectorize_resumes_multiple_inputs(self):
        texts = ["first resume", "second resume"]
        features = extract_tfidf_features_from_resumes(texts)
        self.assertEqual(features.shape[0], 2)

    def test_vectorize_resumes_output_shape(self):
        texts = ["resume A", "resume B"]
        features = extract_tfidf_features_from_resumes(texts)
        self.assertEqual(len(features.shape), 2)  # Should be 2D array

    # ---------- Tests for embed_resumes_with_sbert ----------
    def test_embed_resumes_with_sbert_returns_dataframe(self):
        resumes = ["Resume 1 text", "Resume 2 text"]
        embedding_df = extract_embeddings_from_resumes(resumes)
        self.assertIsInstance(embedding_df, pd.DataFrame)

    def test_embed_resumes_with_sbert_row_count_matches(self):
        resumes = ["Resume 1 text", "Resume 2 text"]
        embedding_df = extract_embeddings_from_resumes(resumes)
        self.assertEqual(embedding_df.shape[0], len(resumes))

    def test_embed_resumes_with_sbert_has_embedding_dimensions(self):
        resumes = ["Sample text"]
        embedding_df = extract_embeddings_from_resumes(resumes)
        self.assertGreater(embedding_df.shape[1], 0)

    def test_embed_resumes_with_sbert_column_dtype_is_float(self):
        resumes = ["AI and ML resume"]
        embedding_df = extract_embeddings_from_resumes(resumes)
        self.assertTrue(all(dtype.kind == 'f' for dtype in embedding_df.dtypes))

