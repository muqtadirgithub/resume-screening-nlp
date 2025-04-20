import unittest
from preprocessing.preprocessing import preprocess_resume

class TestPreprocessing(unittest.TestCase):
    
    def test_removes_punctuation_and_lowercases(self):
        self.assertEqual(preprocess_resume("Hello!!!"), "hello")

    def test_handles_empty_input(self):
        self.assertEqual(preprocess_resume(""), "")

    def test_removes_stopwords(self):
        self.assertEqual(preprocess_resume("This is an example of a sentence"), "example sentence")

    def test_applies_lemmatization(self):
        self.assertEqual(preprocess_resume("The cats are running and eating"), "cat running eating")

    def test_combines_all_cleaning_steps(self):
        text = "!!!   The DATA scientists were analyzing the datasets in 2023.   "
        self.assertEqual(preprocess_resume(text), "data scientist analyzing datasets")

    def test_handles_special_characters(self):
        self.assertEqual(preprocess_resume("Machine@Learning #1!"), "machinelearning")

    def test_handles_mixed_case_and_stopwords(self):
        self.assertEqual(preprocess_resume("WE are Learning PYTHON for DATA Science"), "learning python data science")