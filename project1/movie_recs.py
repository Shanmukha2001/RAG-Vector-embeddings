from dotenv import load_dotenv
import os
import pymongo
from sentence_transformers import SentenceTransformer

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

query = "imaginary characters from outer space at war"

# MongoDB Aggregation Query
pipeline = [
    {
        "$search": {
            "index": "plotSemanticSearchIndex",  # Use the name of your index
            "knnBeta": {
                "vector": generate_embedding(query),  # The vector to search
                "path": "plot_embedding_hf",  # The field in the index
                "k": 4  # Number of nearest neighbors to retrieve
            }
        }
    }
]

# Run the aggregation pipeline
try:
    results = collection.aggregate(pipeline)
    for document in results:
        print(f"Movie Name: {document.get('title', 'Unknown')}")
        print(f"Score: {document.get('_score', 'N/A')}")
        print(f"Plot: {document.get('plot', 'No plot available')}\n")
except Exception as e:
    print(f"Error occurred: {e}")
