# app/vectorstore.py
from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialization (run once at app startup)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.get_or_create_collection(name="genetics_faqs")

def initialize_vectorstore():
    """Populate the collection with genetics knowledge chunks if not already populated."""
    # Check if collection already has data
    if collection.count() > 0:
        logger.info(f"Vectorstore already initialized with {collection.count()} documents")
        return
    
    # Sample genetics knowledge base (expand this significantly)
    genetics_knowledge = [
        {
            "text": "What is gene therapy? Gene therapy is a medical technique that uses genes to treat or prevent disease. It works by introducing genetic material into cells to compensate for abnormal genes or to make a beneficial protein.",
            "source": "NIH Genetics Home Reference",
            "category": "gene_therapy"
        },
        {
            "text": "What does pathogenic variant mean? A pathogenic variant is a genetic change that is known to cause disease or increase the risk of developing a disease. These variants are typically classified as disease-causing based on scientific evidence.",
            "source": "ClinVar Database",
            "category": "variant_classification"
        },
        {
            "text": "What is autosomal dominant inheritance? Autosomal dominant inheritance means that a single copy of the altered gene in each cell is sufficient to cause the disorder. Each child of an affected parent has a 50% chance of inheriting the altered gene.",
            "source": "Genetics Textbook",
            "category": "inheritance_patterns"
        },
        {
            "text": "What is a carrier? A carrier is a person who has one copy of a gene mutation for a recessive disorder but does not have the disorder themselves. Carriers can pass the mutation to their children.",
            "source": "Genetics Education",
            "category": "inheritance_patterns"
        },
        {
            "text": "What is genetic counseling? Genetic counseling is a process that helps individuals and families understand genetic information and its implications for health and reproduction. It involves education, risk assessment, and support.",
            "source": "NSGC Guidelines",
            "category": "genetic_counseling"
        },
        {
            "text": "What is a chromosome? A chromosome is a structure found in the nucleus of cells that contains DNA. Humans have 23 pairs of chromosomes, with one set inherited from each parent.",
            "source": "Basic Genetics",
            "category": "basic_concepts"
        },
        {
            "text": "What is DNA? DNA (deoxyribonucleic acid) is the hereditary material in humans and almost all other organisms. It contains the instructions needed for an organism to develop, survive, and reproduce.",
            "source": "NIH Genetics",
            "category": "basic_concepts"
        },
        {
            "text": "What is a mutation? A mutation is a change in the DNA sequence. Mutations can be harmful, beneficial, or neutral, depending on their effect on the organism and the environment.",
            "source": "Genetics Reference",
            "category": "basic_concepts"
        }
    ]
    
    # Process and add documents to vectorstore
    documents = []
    embeddings = []
    ids = []
    metadatas = []
    
    for idx, knowledge in enumerate(genetics_knowledge):
        documents.append(knowledge["text"])
        embeddings.append(embedding_model.encode(knowledge["text"]).tolist())
        ids.append(str(idx))
        metadatas.append({
            "source": knowledge["source"],
            "category": knowledge["category"]
        })
    
    # Add to collection
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    logger.info(f"Initialized vectorstore with {len(documents)} genetics knowledge documents")

def retrieve_similar_chunks(query: str, n_results: int = 3, category_filter: Optional[str] = None) -> List[Dict]:
    """Retrieve most similar genetics knowledge chunks for a query text."""
    try:
        query_embedding = embedding_model.encode(query).tolist()
        
        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        
        # Add category filter if specified
        if category_filter:
            query_params["where"] = {"category": category_filter}
        
        results = collection.query(**query_params)
        
        # Format results
        retrieved_chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                chunk = {
                    "text": doc,
                    "source": results["metadatas"][0][i]["source"],
                    "category": results["metadatas"][0][i]["category"],
                    "similarity_score": results["distances"][0][i] if "distances" in results else None
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    except Exception as e:
        logger.error(f"Error retrieving similar chunks: {e}")
        return []

def add_knowledge_chunk(text: str, source: str, category: str) -> bool:
    """Add a new knowledge chunk to the vectorstore."""
    try:
        # Generate embedding
        embedding = embedding_model.encode(text).tolist()
        
        # Get next available ID
        next_id = str(collection.count())
        
        # Add to collection
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[next_id],
            metadatas=[{"source": source, "category": category}]
        )
        
        logger.info(f"Added new knowledge chunk: {text[:50]}...")
        return True
    
    except Exception as e:
        logger.error(f"Error adding knowledge chunk: {e}")
        return False

def get_collection_stats() -> Dict:
    """Get statistics about the vectorstore collection."""
    try:
        count = collection.count()
        return {
            "total_documents": count,
            "collection_name": collection.name
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {"error": str(e)}

# Initialize on import
initialize_vectorstore()
