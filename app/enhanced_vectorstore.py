"""
Enhanced vectorstore module for genetic counseling RAG system.
Integrates with data processor and provides advanced retrieval capabilities.
"""

import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional, Any
import os
from pathlib import Path
import json
from datetime import datetime

from .data_processor import DataProcessor

logger = logging.getLogger(__name__)

class EnhancedVectorStore:
    """Enhanced vectorstore with advanced retrieval and management capabilities."""
    
    def __init__(self, 
                 collection_name: str = "genetic_counseling_kb",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "data/chroma_db"):
        """
        Initialize the enhanced vectorstore.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model name
            persist_directory: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Genetic counseling knowledge base"}
        )
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        logger.info(f"Enhanced vectorstore initialized with collection: {collection_name}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand content types
            sample_results = self.collection.get(limit=min(100, count))
            categories = {}
            sources = {}
            types = {}
            
            if sample_results and sample_results.get('metadatas'):
                for metadata in sample_results['metadatas']:
                    cat = metadata.get('category', 'unknown')
                    src = metadata.get('source', 'unknown')
                    typ = metadata.get('type', 'unknown')
                    
                    categories[cat] = categories.get(cat, 0) + 1
                    sources[src] = sources.get(src, 0) + 1
                    types[typ] = types.get(typ, 0) + 1
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "categories": categories,
                "sources": sources,
                "types": types,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Add multiple chunks to the vectorstore.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
            batch_size: Number of chunks to process at once
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks:
                logger.warning("No chunks to add")
                return False
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare batch data
                documents = []
                embeddings = []
                ids = []
                metadatas = []
                
                for j, chunk in enumerate(batch):
                    if not chunk.get('text', '').strip():
                        continue
                    
                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk['text']).tolist()
                    
                    # Prepare metadata
                    metadata = {
                        'source': chunk.get('source', 'unknown'),
                        'category': chunk.get('category', 'general'),
                        'type': chunk.get('type', 'text'),
                        'added_at': datetime.now().isoformat()
                    }
                    
                    # Add any additional metadata
                    for key, value in chunk.items():
                        if key not in ['text', 'source', 'category', 'type'] and value is not None:
                            metadata[key] = str(value)
                    
                    documents.append(chunk['text'])
                    embeddings.append(embedding)
                    ids.append(f"chunk_{i + j}_{datetime.now().timestamp()}")
                    metadatas.append(metadata)
                
                # Add batch to collection
                if documents:
                    self.collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas
                    )
                    logger.info(f"Added batch of {len(documents)} chunks")
            
            logger.info(f"Successfully added {len(chunks)} chunks to vectorstore")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            return False
    
    def retrieve_similar_chunks(self, 
                               query: str, 
                               n_results: int = 5,
                               category_filter: Optional[str] = None,
                               source_filter: Optional[str] = None,
                               type_filter: Optional[str] = None,
                               min_similarity: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar chunks with advanced filtering options.
        
        Args:
            query: Search query
            n_results: Number of results to return
            category_filter: Filter by category
            source_filter: Filter by source
            type_filter: Filter by type
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build where clause for filtering
            where_clause = {}
            if category_filter:
                where_clause['category'] = category_filter
            if source_filter:
                where_clause['source'] = source_filter
            if type_filter:
                where_clause['type'] = type_filter
            
            # Query the collection
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            
            if where_clause:
                query_params["where"] = where_clause
            
            results = self.collection.query(**query_params)
            
            # Process results
            retrieved_chunks = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    # Calculate similarity score (1 - distance for cosine similarity)
                    similarity_score = 1 - results["distances"][0][i] if "distances" in results else None
                    
                    # Apply similarity threshold if specified
                    if min_similarity and similarity_score and similarity_score < min_similarity:
                        continue
                    
                    chunk = {
                        "text": doc,
                        "source": results["metadatas"][0][i].get("source", "unknown"),
                        "category": results["metadatas"][0][i].get("category", "general"),
                        "type": results["metadatas"][0][i].get("type", "text"),
                        "similarity_score": similarity_score,
                        "metadata": results["metadatas"][0][i]
                    }
                    retrieved_chunks.append(chunk)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query[:50]}...")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving similar chunks: {e}")
            return []
    
    def search_by_metadata(self, 
                          category: Optional[str] = None,
                          source: Optional[str] = None,
                          type: Optional[str] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search chunks by metadata filters.
        
        Args:
            category: Filter by category
            source: Filter by source
            type: Filter by type
            limit: Maximum number of results
            
        Returns:
            List of matching chunks
        """
        try:
            where_clause = {}
            if category:
                where_clause['category'] = category
            if source:
                where_clause['source'] = source
            if type:
                where_clause['type'] = type
            
            results = self.collection.get(
                where=where_clause if where_clause else None,
                limit=limit
            )
            
            chunks = []
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents']):
                    chunk = {
                        "text": doc,
                        "source": results['metadatas'][i].get("source", "unknown"),
                        "category": results['metadatas'][i].get("category", "general"),
                        "type": results['metadatas'][i].get("type", "text"),
                        "metadata": results['metadatas'][i]
                    }
                    chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} chunks matching metadata filters")
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def initialize_from_csv_files(self, csv_files: Dict[str, str]) -> bool:
        """
        Initialize the vectorstore with data from CSV files.
        
        Args:
            csv_files: Dictionary mapping file types to file paths
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if collection already has data
            if self.collection.count() > 0:
                logger.info(f"Collection already has {self.collection.count()} documents. Skipping initialization.")
                return True
            
            # Process CSV files
            chunks = self.data_processor.process_all_csv_files(csv_files)
            
            if not chunks:
                logger.warning("No chunks generated from CSV files")
                return False
            
            # Add chunks to vectorstore
            success = self.add_chunks(chunks)
            
            if success:
                logger.info(f"Successfully initialized vectorstore with {len(chunks)} chunks")
                # Save processed data for backup
                self.data_processor.save_processed_data(chunks)
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing from CSV files: {e}")
            return False
    
    def add_document(self, 
                    text: str, 
                    source: str, 
                    category: str = "general",
                    **metadata) -> bool:
        """
        Add a single document to the vectorstore.
        
        Args:
            text: Document text
            source: Source of the document
            category: Category of the document
            **metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            chunk = {
                'text': text,
                'source': source,
                'category': category,
                'type': 'document',
                **metadata
            }
            
            return self.add_chunks([chunk])
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def delete_chunks_by_source(self, source: str) -> bool:
        """
        Delete all chunks from a specific source.
        
        Args:
            source: Source name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all chunks from the source
            chunks = self.search_by_metadata(source=source)
            
            if not chunks:
                logger.info(f"No chunks found for source: {source}")
                return True
            
            # Extract IDs (this is a limitation of ChromaDB - we need to track IDs)
            # For now, we'll use a workaround by recreating the collection
            logger.warning("Deleting chunks by source requires collection recreation. This is a ChromaDB limitation.")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting chunks by source: {e}")
            return False
    
    def export_collection(self, output_file: str) -> bool:
        """
        Export the entire collection to a JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents
            results = self.collection.get()
            
            if not results or not results.get('documents'):
                logger.warning("No documents to export")
                return False
            
            # Prepare export data
            export_data = []
            for i, doc in enumerate(results['documents']):
                export_data.append({
                    'text': doc,
                    'metadata': results['metadatas'][i] if results.get('metadatas') else {},
                    'id': results['ids'][i] if results.get('ids') else f"doc_{i}"
                })
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(export_data)} documents to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return False

# Global instance for easy access
_vectorstore_instance = None

def get_vectorstore() -> EnhancedVectorStore:
    """Get the global vectorstore instance."""
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = EnhancedVectorStore()
    return _vectorstore_instance

def initialize_vectorstore(csv_files: Optional[Dict[str, str]] = None) -> bool:
    """
    Initialize the global vectorstore instance.
    
    Args:
        csv_files: Optional CSV files to initialize with
        
    Returns:
        True if successful, False otherwise
    """
    global _vectorstore_instance
    _vectorstore_instance = EnhancedVectorStore()
    
    if csv_files:
        return _vectorstore_instance.initialize_from_csv_files(csv_files)
    
    return True
