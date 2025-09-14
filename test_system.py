#!/usr/bin/env python3
"""
Test script for the Genetic Counseling RAG System.
This script tests the system components and provides example queries.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_processing():
    """Test the data processing system."""
    print("Testing data processing...")
    
    try:
        from app.data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Test CSV files
        csv_files = {
            'gccg': 'GCCG.csv',
            'questions': 'Genetic_bot_qb_with_genes_rerun_v3.csv',
            'training_questions': 'genetic_counsellor_training_questions_with_official_sources.csv'
        }
        
        # Check if files exist
        existing_files = {}
        for key, file_path in csv_files.items():
            if os.path.exists(file_path):
                existing_files[key] = file_path
                print(f"✓ Found {key}: {file_path}")
            else:
                print(f"✗ Missing {key}: {file_path}")
        
        if existing_files:
            # Process files
            chunks = processor.process_all_csv_files(existing_files)
            print(f"✓ Processed {len(chunks)} chunks from CSV files")
            
            # Show sample chunks
            if chunks:
                print("\nSample chunks:")
                for i, chunk in enumerate(chunks[:3]):
                    print(f"  {i+1}. {chunk['text'][:100]}...")
                    print(f"     Source: {chunk['source']}, Category: {chunk['category']}")
        else:
            print("✗ No CSV files found for processing")
            
    except Exception as e:
        print(f"✗ Error testing data processing: {e}")

def test_vectorstore():
    """Test the vectorstore system."""
    print("\nTesting vectorstore...")
    
    try:
        from app.enhanced_vectorstore import EnhancedVectorStore
        
        # Initialize vectorstore
        vectorstore = EnhancedVectorStore()
        print("✓ Vectorstore initialized")
        
        # Test adding sample data
        sample_chunks = [
            {
                'text': 'What is gene therapy? Gene therapy is a medical technique that uses genes to treat or prevent disease.',
                'source': 'Test Source',
                'category': 'test',
                'type': 'definition'
            },
            {
                'text': 'What is a mutation? A mutation is a change in the DNA sequence.',
                'source': 'Test Source',
                'category': 'test',
                'type': 'definition'
            }
        ]
        
        success = vectorstore.add_chunks(sample_chunks)
        if success:
            print("✓ Sample data added to vectorstore")
        else:
            print("✗ Failed to add sample data")
        
        # Test retrieval
        results = vectorstore.retrieve_similar_chunks("What is gene therapy?", n_results=2)
        if results:
            print(f"✓ Retrieved {len(results)} similar chunks")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['text'][:50]}... (similarity: {result.get('similarity_score', 'N/A')})")
        else:
            print("✗ No results retrieved")
        
        # Test stats
        stats = vectorstore.get_collection_stats()
        print(f"✓ Collection stats: {stats.get('total_documents', 0)} documents")
        
    except Exception as e:
        print(f"✗ Error testing vectorstore: {e}")

def test_rag_system():
    """Test the complete RAG system."""
    print("\nTesting RAG system...")
    
    try:
        from app.enhanced_vectorstore import get_vectorstore
        from app.data_processor import DataProcessor
        
        # Initialize vectorstore
        vectorstore = get_vectorstore()
        
        # Process and add CSV data
        processor = DataProcessor()
        csv_files = {
            'gccg': 'GCCG.csv',
            'questions': 'Genetic_bot_qb_with_genes_rerun_v3.csv',
            'training_questions': 'genetic_counsellor_training_questions_with_official_sources.csv'
        }
        
        # Check existing files
        existing_files = {k: v for k, v in csv_files.items() if os.path.exists(v)}
        
        if existing_files:
            chunks = processor.process_all_csv_files(existing_files)
            if chunks:
                success = vectorstore.add_chunks(chunks)
                if success:
                    print(f"✓ Added {len(chunks)} chunks to vectorstore")
                else:
                    print("✗ Failed to add chunks to vectorstore")
                    return
            else:
                print("✗ No chunks processed from CSV files")
                return
        else:
            print("✗ No CSV files found")
            return
        
        # Test queries
        test_queries = [
            "What is gene therapy?",
            "What does pathogenic variant mean?",
            "How is Huntington's disease inherited?",
            "What is genetic counseling?",
            "What are the risks of amniocentesis?"
        ]
        
        print("\nTesting queries:")
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = vectorstore.retrieve_similar_chunks(query, n_results=3)
            if results:
                print(f"  Retrieved {len(results)} results:")
                for i, result in enumerate(results):
                    print(f"    {i+1}. {result['text'][:80]}...")
                    print(f"       Source: {result['source']}, Score: {result.get('similarity_score', 'N/A')}")
            else:
                print("  No results found")
        
    except Exception as e:
        print(f"✗ Error testing RAG system: {e}")

def test_api_endpoints():
    """Test the API endpoints (requires server to be running)."""
    print("\nTesting API endpoints...")
    
    try:
        import requests
        import time
        
        # Wait a moment for server to start
        time.sleep(2)
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Health endpoint working")
                health_data = response.json()
                print(f"  Status: {health_data.get('status')}")
                print(f"  Documents: {health_data.get('vectorstore', {}).get('documents', 0)}")
            else:
                print(f"✗ Health endpoint returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to server. Make sure it's running on localhost:8000")
            return
        
        # Test chat endpoint
        try:
            chat_data = {
                "message": "What is gene therapy?",
                "filters": {}
            }
            response = requests.post(f"{base_url}/chat", json=chat_data, timeout=10)
            if response.status_code == 200:
                print("✓ Chat endpoint working")
                chat_response = response.json()
                print(f"  Reply: {chat_response.get('reply', '')[:100]}...")
                print(f"  Sources: {chat_response.get('sources', [])}")
            else:
                print(f"✗ Chat endpoint returned status {response.status_code}")
        except Exception as e:
            print(f"✗ Error testing chat endpoint: {e}")
        
        # Test knowledge stats
        try:
            response = requests.get(f"{base_url}/knowledge/stats", timeout=5)
            if response.status_code == 200:
                print("✓ Knowledge stats endpoint working")
                stats = response.json()
                print(f"  Total documents: {stats.get('total_documents', 0)}")
            else:
                print(f"✗ Knowledge stats endpoint returned status {response.status_code}")
        except Exception as e:
            print(f"✗ Error testing knowledge stats: {e}")
            
    except ImportError:
        print("✗ Requests library not available. Install with: pip install requests")
    except Exception as e:
        print(f"✗ Error testing API endpoints: {e}")

def main():
    """Main test function."""
    print("Genetic Counseling RAG System - Test Suite")
    print("=" * 50)
    
    # Test individual components
    test_data_processing()
    test_vectorstore()
    test_rag_system()
    
    print("\n" + "=" * 50)
    print("Component tests completed.")
    print("\nTo test the full API, run the server and then test API endpoints:")
    print("1. Start server: python run_server.py")
    print("2. Test API: python test_system.py --api")
    
    # Check if API testing is requested
    if "--api" in sys.argv:
        test_api_endpoints()

if __name__ == "__main__":
    main()
