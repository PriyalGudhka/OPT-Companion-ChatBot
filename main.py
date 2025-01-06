from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import os
from dotenv import load_dotenv

load_dotenv()

st.title("OPT Companion AI Bot")

# Initialize session state for responses and requests if not already present
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize OpenAI LLM for the chatbot using GPT-3.5
llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize memory for the conversation, using a buffer window with the last 3 messages
if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are an assistant specifically designed to help students navigate the complexities of OPT (Optional Practical Training). You have a friendly and professional persona, always focusing on providing clear and accurate information about OPT, visa regulations, work eligibility, and related documentation. If the question falls outside your knowledge, say 'I don't know.' Answer the question truthfully using the context provided.'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Containers for displaying chat history and user input
response_container = st.container()
textcontainer = st.container()

# Handle the user input and response generation
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query) 
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}") 
        # Append the user's query and the assistant's response to the session state for history
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

# Display the chat history (responses and user queries)
with response_container:
    if st.session_state['responses']:
        # Iterate through the responses and display them along with the corresponding user queries
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')