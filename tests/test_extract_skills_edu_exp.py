import unittest
from webapp.app import extract_skills_edu_exp

class TestSkillEduExpExtraction(unittest.TestCase):

    def test_extraction_detects_skills_edu_exp(self):
        resume_text = """
        Jane Doe has a Master's degree in Computer Science.
        She worked as a data scientist and is skilled in Python, TensorFlow, and SQL.
        Also experienced in cloud platforms like AWS.
        """

        extracted = extract_skills_edu_exp(resume_text)

        self.assertTrue(any("computer science" in line.lower() for line in extracted), "Education not detected")
        self.assertTrue(any("python" in line.lower() for line in extracted), "Skill not detected")
        self.assertTrue(any("worked" in line.lower() for line in extracted), "Experience not detected")
