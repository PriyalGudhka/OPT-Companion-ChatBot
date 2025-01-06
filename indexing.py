# Import necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize Pinecone client with API key from environment variable
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "opt-chatbot"

# Check if the index exists; if not, create it
if 'opt-chatbot' not in pc.list_indexes().names():
    pc.create_index(
        name='opt-chatbot',
        dimension=384,  # The dimensionality of the embeddings
        metric='euclidean',  # The distance metric used to compare embeddings
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = "opt-chatbot"

# Function to load and prepare dataset
def load_and_prepare_dataset():
    # Load dataset from text files
    dataset = load_dataset("text", data_files="data/*.txt")
    
    dataset = dataset["train"].map(lambda x: {"label": 0})
    
    # Split data into train and validation sets (80% train, 20% validation)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset["text"], dataset["label"], test_size=0.2, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels

# Load and prepare the dataset
train_texts, val_texts, train_labels, val_labels = load_and_prepare_dataset()

# Function to tokenize text data
def tokenize_data(texts, tokenizer, max_length=512):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length)

# Load pre-trained BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the training and validation texts
train_encodings = tokenize_data(train_texts, tokenizer)
val_encodings = tokenize_data(val_texts, tokenizer)

# Define custom Dataset class to handle tokenized data
class OPTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Convert tokenized data into Dataset objects for PyTorch
train_dataset = OPTDataset(train_encodings, train_labels)
val_dataset = OPTDataset(val_encodings, val_labels)

# Load the pre-trained BERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up training arguments (hyperparameters)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize the Trainer class for fine-tuning the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model using the training data
trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Evaluate the fine-tuned model on the validation dataset
results = trainer.evaluate()
print("Evaluation Results:", results)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get embeddings for a list of texts using SentenceTransformer
def get_embeddings(texts, model):
    return model.encode(texts, convert_to_numpy=True)

index_name = pc.Index('opt-chatbot')

# Function to index the embeddings in Pinecone
def index_in_pinecone(texts, embeddings, index):
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        index_name.upsert([(str(i), embedding.tolist(), {"text": text})])

train_embeddings = get_embeddings(train_texts[:100], model)

index_in_pinecone(train_texts[:100], train_embeddings, index_name)

# Function to query Pinecone index with a user query
def query_pinecone(query, model, tokenizer, index, top_k=5):
    query_embedding = get_embeddings([query], model)[0]

    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)

    return results

# Example query to test the system
query = "What is OPT?"
results = query_pinecone(query, model, tokenizer, index_name)

print("Query Results:")
for match in results["matches"]:
    print(f"Score: {match['score']}, Text: {match['metadata']['text']}")