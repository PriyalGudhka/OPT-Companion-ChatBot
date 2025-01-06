from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load pre-trained Sentence Transformer model

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# Create the Pinecone index if it does not exist
if 'opt-chatbot' not in pc.list_indexes().names():
    pc.create_index(
        name='opt-chatbot',
        dimension=384,  # Dimensionality of the embedding vector
        metric='euclidean',  # Metric used for similarity search
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Access the 'opt-chatbot' index
index = pc.Index('opt-chatbot')   

# Function to find the most similar matches from Pinecone index for a given input text
def find_match(input_text):
    input_em = model.encode(input_text).tolist()  # Encode the input text into an embedding vector
    
    # Query Pinecone index for the top 2 similar matches
    result = index.query(
        vector=input_em,
        top_k=2, 
        include_metadata=True  # Retrieve metadata along with similarity score
    )
    
    if len(result['matches']) < 2:
        return "Not enough matches found."
    
    return (
        result['matches'][0]['metadata']['text'] + "\n" +
        result['matches'][1]['metadata']['text']
    )

# Function to refine a user query using OpenAI's GPT model
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[ 
        {"role": "system", "content": "You are a helpful assistant for refining user queries."},
        {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
    ],
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
    return response['choices'][0]['message']['content']  # Return the refined query from the model's response

# Function to generate a conversation string from session state for context
def get_conversation_string():
    conversation_string = ""
    # Construct the conversation string using previous requests and responses
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string