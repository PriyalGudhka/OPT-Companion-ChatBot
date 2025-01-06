import unittest
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import os
from dotenv import load_dotenv
from langchain.chains.conversation.memory import ConversationBufferMemory

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Prepare the system and human message templates
system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are an assistant specifically designed to help students navigate the complexities of OPT (Optional Practical Training). You have a friendly and professional persona, always focusing on providing clear and accurate information about OPT, visa regulations, work eligibility, and related documentation. If the question falls outside your knowledge or is about an unrelated topic (e.g., weather, general knowledge), say 'I don't know.' Answer the question truthfully using the context provided.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Initialize memory (using ConversationBufferMemory for better testing behavior)
buffer_memory = ConversationBufferMemory(k=3, return_messages=True)

# Initialize the conversation chain with memory
conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Test function to simulate the conversation
def test_chatbot(query, expected_response_partial):
    print(f"Testing query: {query}")
    response = conversation.predict(input=f"Query: {query}")
    print(f"Chatbot Response: {response}")
    
    # Assertion to check if the expected partial response is in the actual response
    assert expected_response_partial in response, f"Test failed for query: {query}. Expected partial match: '{expected_response_partial}', but got: '{response}'"

class TestChatbot(unittest.TestCase):

    def test_opt_eligibility(self):
        # Test 1: General OPT question
        test_chatbot("What is OPT?", "Optional Practical Training")

    def test_weather_query(self):
        # Test 2: Invalid query (weather-related question)
        test_chatbot("What is the weather like today?", "I don't know.")

if __name__ == "__main__":
    unittest.main()
