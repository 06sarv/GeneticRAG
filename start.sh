#!/bin/bash

# GeneticRAG System Startup Script

echo "Starting GeneticRAG system..."

# check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

# activate virtual environment
source venv/bin/activate

# check if required files exist
if [ ! -f "data/raw/GCCG.csv" ] || [ ! -f "data/raw/Genetic_bot_qb_with_genes_rerun_v3.csv" ] || [ ! -f "data/raw/genetic_counsellor_training_questions_with_official_sources.csv" ]; then
    echo "Warning: Some CSV files are missing. The system will work with available files."
fi

# start the server
echo "Starting server on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

python3 run_server.py
