# OPT Companion: AI-Powered Work Authorization Assistant

## 1 About
A chatbot system designed to answer questions related to the OPT (Optional Practical Training) process. It uses a pre-trained model, fine-tuned for sequence classification, to understand the context of user queries. The system also incorporates Pinecone, a vector database, for fast and efficient retrieval of relevant answers from a large dataset.


## 2. Project Architecture
The architecture of the OPT chatbot consists of multiple components working together to provide a seamless and responsive user experience.

![System Architecture Diagram](./images/System%20Architecture%20Diagram.png)



## 3. Project Setup

1. Create a .env file with the following environment variables:

    1. OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
    2. PINECONE_API_KEY==<YOUR_OPENAI_API_KEY>

2. Steps to run the application

    **Step 1:** pip install -r requirements.txt
    **Step 2:**  python indexing.py // This will create index in pinecone
    **Step 3:**  python -m streamlit run main.py
    **Step 4:**  python -m streamlit run main.py

3. To run test files

    **Step 1:** python tests/test_utils.py
    **Step 2:**  python tests/test_main.py
    **Step 3:**  python tests/test_indexing.py

4. You will be able to access the application at: http://localhost:8501/

## 4. Conversations
![Conversation1 ](./Example%20conversations/Conversation1.jpeg)
![Conversation2 ](./Example%20conversations/Conversation2.jpeg)
![Conversation3 ](./Example%20conversations/Conversation3.jpeg)
![Conversation4 ](./Example%20conversations/ErrorCase.jpeg)


## 5. Implementation Details
 
### **Prompt Engineering**
To enhance the chatbot’s ability to understand and respond effectively, systematic prompting strategies were employed:
- **Persona Pattern**: The model is using Assistant persona pattern to assist the students.
- **Context Management**: The chatbot maintains the context of ongoing conversations to ensure continuity in multi-turn interactions. This allows the chatbot to refer to previous messages and make conversations feel natural.
- **Specialized Conversation Flows**: Custom conversation flows were designed for common user queries related to OPT (Optional Practical Training), ensuring that the chatbot provides accurate and relevant information.
- **Error Handling**: The chatbot gracefully handles errors by prompting the user to rephrase their query or offering guidance on how to obtain further help.

### **Fine-Tuning**
The core fine-tuning process involves training a pre-trained language model on a domain-specific dataset to improve its performance on OPT-related tasks. This includes:

- **Domain-Specific Dataset**: Queries and responses specific to OPT help the model understand the nuances of the domain.
- **Training Process**: Fine-tuning is performed for several epochs, optimizing the model for accurate responses.
- **Evaluation Metrics**: Performance is evaluated using accuracy, F1 score, and loss metrics to ensure improvements.




### **Retrieval-Augmented Generation (RAG)**
RAG enhances responses by retrieving domain-specific documents and integrating them into the model's output. Steps involved:

- **Knowledge Base**: Contains detailed documents related to OPT, such as visa regulations and application procedures.
- **Vector Storage and Retrieval**: Uses vector databases Pinecone for quick retrieval of relevant documents.
- **Document Chunking**: Documents are split into smaller, contextually meaningful chunks.
- **Ranking and Filtering**: Retrieves the most relevant documents and filters out less useful information.


## 6. Challenges and Solutions

### **Challenges**

1. **Handling Ambiguity in User Queries**
    - Users often ask vague or ambiguous questions.
    - **Solution**: Context management and clarifying questions narrow down user intent.

2. **Domain-Specific Knowledge**
    - Lack of detailed knowledge can lead to incorrect answers.
    - **Solution**: Fine-tuning on OPT-specific data and using RAG for additional retrieval.

3. **Scalability**
    - The system must handle increased traffic as the user base grows.
    - **Solution**: Docker and Kubernetes enable containerization and efficient deployment.

4. **Error Handling**
    - The chatbot may fail to understand complex queries.
    - **Solution**: A fallback mechanism directs users to human support or suggests rephrasing.

---

## 7. Future Improvements

1. **Enhanced Fine-Tuning**: Continuously improve the model using new data and feedback.
2. **Multilingual Support**: Extend the chatbot’s capabilities to support multiple languages.
3. **Personalization**: Implement user-specific profiles for customized responses.
4. **Advanced RAG**: Improve retrieval and ranking mechanisms for complex queries.
5. **User Feedback Integration**: Develop sophisticated feedback systems to enhance chatbot responses over time.

---

## Conclusion

The OPT chatbot is designed to efficiently handle user queries related to OPT using advanced NLP techniques, fine-tuning, and retrieval-augmented generation. Its architecture ensures accurate and contextually relevant information, while continuous improvements ensure the chatbot evolves with user needs.

By addressing challenges, incorporating performance metrics, and planning for future improvements, the OPT chatbot aims to provide exceptional support to users navigating the OPT process.
## References
Medium Article: https://medium.com/@abonia/document-based-llm-powered-chatbot-bb316009de93
