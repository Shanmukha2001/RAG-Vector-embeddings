import pymongo
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Accessing the MongoDB URI from the .env file
mongo_uri = os.getenv("MONGO_URI")

# Accessing the Sentence Transformer model name from the .env file
sentence_transformer_model = os.getenv("SENTENCE_TRANSFORMER_MODEL")

# MongoDB connection setup
client = pymongo.MongoClient(mongo_uri)
db = client.sample_mflix
collection = db.movies

# Load the model locally (this only needs to be done once)
model = SentenceTransformer(sentence_transformer_model)

def generate_embedding(text: str) -> list:
    """
    Generate embedding using the locally loaded model.
    """
    return model.encode([text]).tolist()[0]  # Ensure it returns a single vector

# Generate embeddings for all documents
for doc in collection.find({'plot': {"$exists": True}}):  # Adjust the limit as necessary
    print(f"Processing movie: {doc['title']}")
    # Generate embedding for the plot of the movie
    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
    # Update the document with the new embedding
    collection.replace_one({'_id': doc['_id']}, doc)

print("Embedding generation and update complete.")
