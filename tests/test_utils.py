import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import unittest
from unittest.mock import patch
import utils  # Assuming utils.py is in the same directory as this test file

class TestUtils(unittest.TestCase):

    # Test the 'find_match' function with a simple question related to OPT
    @patch('utils.index.query')
    def test_find_match(self, mock_query):
        # Setup mock response from Pinecone query
        mock_query.return_value = {
            'matches': [
                {'metadata': {'text': 'OPT is an AI-powered chatbot designed for query refinement and knowledge retrieval.'}},
                {'metadata': {'text': 'OPT helps users with information related to API usage and integration.'}}
            ]
        }
        
        input_text = "What is OPT?"
        result = utils.find_match(input_text)
        
        expected_result = (
            "OPT is an AI-powered chatbot designed for query refinement and knowledge retrieval.\n"
            "OPT helps users with information related to API usage and integration."
        )
        self.assertEqual(result, expected_result)

    # Test the 'query_refiner' function with a simple query about OPT
    @patch('openai.ChatCompletion.create')
    def test_query_refiner(self, mock_create):
        # Setup mock response from OpenAI API
        mock_create.return_value = {
            'choices': [{'message': {'content': 'What is OPT used for?'}}]
        }
        
        conversation = "User: What can I do with OPT?"
        query = "Tell me about OPT"
        result = utils.query_refiner(conversation, query)
        
        expected_result = "What is OPT used for?"
        self.assertEqual(result, expected_result)

    @patch('streamlit.session_state', new_callable=lambda: {
    'requests': ['What is OPT?'],
    'responses': [
        '', 
        "OPT stands for Optional Practical Training, which is a temporary employment authorization for international students in the United States on an F-1 visa. It allows students to gain practical work experience directly related to their field of study."
    ]
})
    def test_get_conversation_string(self, mock_session_state):
        result = utils.get_conversation_string()

        expected_conversation = (
        "Human: What is OPT?\n"
        "Bot: OPT stands for Optional Practical Training, which is a temporary employment authorization for international students in the United States on an F-1 visa. It allows students to gain practical work experience directly related to their field of study.\n"
    )
        self.assertEqual(result, expected_conversation)

if __name__ == '__main__':
    unittest.main()
