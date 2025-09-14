#!/bin/bash

# Genetic Counseling RAG System Startup Script

echo "Starting Genetic Counseling RAG System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required files exist
if [ ! -f "GCCG.csv" ] || [ ! -f "Genetic_bot_qb_with_genes_rerun_v3.csv" ] || [ ! -f "genetic_counsellor_training_questions_with_official_sources.csv" ]; then
    echo "Warning: Some CSV files are missing. The system will work with available files."
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

python3 run_server.py
