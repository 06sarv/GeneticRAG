# app/vectorstore.py
from sentence_transformers import SentenceTransformer
import chromadb

# Initialization (run once at app startup)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.get_or_create_collection(name="genetics_faqs")

def initialize_vectorstore():
    """Populate the collection with FAW chunks if not already populated."""
    sample_faqs = [
        {"question": "What is gene therapy?", "answer": "Gene therapy modifies genes to treat disease."},
        {"question": "What does pathogenic variant mean?", "answer": "A pathogenic variant increases disease risk."},
    ]
    chunks = [f"Q: {faq['question']} A: {faq['answer']}" for faq in sample_faqs]
    for idx, text in enumerate(chunks):
        embedding = embedding_model.encode(text).tolist()
        collection.add(documents=[text], embeddings=[embedding], ids=[str(idx)])

def retrieve_similar_chunks(query, n_results=2):
    """Retrieve most similar FAQ chunks for a query text."""
    query_emb = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=n_results)
    docs = results["documents"][0]
    # You may want to provide source or other meta info as needed
    return [{"text": doc, "source": "Genetics FAQ"} for doc in docs]

# Call on startup (don't repopulate on every import!)
initialize_vectorstore()
