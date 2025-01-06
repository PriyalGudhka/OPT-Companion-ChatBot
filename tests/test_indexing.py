import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import unittest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict
from indexing import (
    load_and_prepare_dataset,
    get_embeddings,
    query_pinecone
)
import numpy as np
from datasets import Dataset

class TestIndexingModule(unittest.TestCase):

     @patch("indexing.load_dataset")
     def test_load_and_prepare_dataset(self, mock_load_dataset):
        # Mocking the dataset loading as a DatasetDict object with 'train' key
        mock_load_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict({
                "text": ["This is a test.", "Another test."],
                "label": [0, 0]
            })
        })

        train_texts, val_texts, train_labels, val_labels = load_and_prepare_dataset()

        self.assertEqual(len(train_texts), 1)
        self.assertEqual(len(val_texts), 1)
        self.assertEqual(train_labels, [0])    

     @patch("sentence_transformers.SentenceTransformer")
     def test_get_embeddings(self, mock_sentence_model):
        # Mocking the SentenceTransformer model
        mock_sentence_model.return_value.encode.return_value = np.array([[0.1] * 384])
        
        # Test get_embeddings function
        model = MagicMock()  # Mocking SentenceTransformer model
        model.encode.return_value = np.array([[0.1] * 384])  # Mocked output embeddings

        texts = ["This is a test sentence."]
        embeddings = get_embeddings(texts, model)

        self.assertTrue(np.array_equal(embeddings, np.array([[0.1] * 384])))

     @patch("pinecone.Pinecone")
     @patch("sentence_transformers.SentenceTransformer")
     def test_query_pinecone(self, mock_sentence_model, mock_pinecone):
        # Mock SentenceTransformer model
        mock_sentence_model.return_value.encode.return_value = np.array([[0.1] * 384])

        # Mock Pinecone index
        mock_index = MagicMock()
        mock_pinecone.return_value = mock_index
        mock_index.query.return_value = {"matches": [{"score": 0.95, "metadata": {"text": "This is a test text."}}]}

        # Test query_pinecone function
        query = "What is OPT?"
        results = query_pinecone(query, mock_sentence_model.return_value, None, mock_index)

        self.assertEqual(len(results["matches"]), 1)
        self.assertEqual(results["matches"][0]["metadata"]["text"], "This is a test text.")

if __name__ == "__main__":
    unittest.main()
