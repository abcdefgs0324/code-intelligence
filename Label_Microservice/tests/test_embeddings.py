import unittest
from unittest.mock import Mock
import os
import code_intelligence
from code_intelligence.embeddings import detect_duplicate_issues
from code_intelligence.embeddings import cosine_similarity

class TestEmbeddings(unittest.TestCase):

    def test_detect_duplicate_issues(self):
        """Testing detect_duplicate_issues function while threshold is 0.99"""
        issue_embedding = [1.0, 1.0]
        all_issue_embeddings = [[1.0, 0], [0.5, 0.8], [1.0, 0.9]]
        duplicate_list = detect_duplicate_issues(cosine_threshold=0.99,
                                                 issue_embedding=issue_embedding,
                                                 all_issue_embeddings=all_issue_embeddings)
        # [(index, cosine_similarity)]
        assert duplicate_list == [(2, cosine_similarity(issue_embedding, all_issue_embeddings[2]))]
