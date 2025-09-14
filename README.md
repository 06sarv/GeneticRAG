# GeneticRAG

A modular Retrieval-Augmented Generation (RAG) system for answering genetic counseling questions using local processing and vector search.

## Features

- Modular architecture for easy integration with other modules
- Local processing without external API dependencies
- Support for multiple data sources (CSV, text, Markdown, JSON, YAML)
- Advanced vector search using ChromaDB with sentence transformers
- Flexible LLM integration with vLLM and transformers support
- Comprehensive configuration with YAML and environment variables
- RESTful API built with FastAPI
- Document ingestion system for easy data source addition

## Project Structure

```
genetic_chatbot_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # fastapi application
│   ├── config.py              # configuration management
│   ├── data_processor.py      # csv and data processing
│   ├── enhanced_vectorstore.py # vector database operations
│   ├── document_ingestion.py  # document ingestion system
│   └── vectorstore.py         # legacy vectorstore (deprecated)
├── data/                      # data directory
│   └── raw/                   # raw csv files
├── scripts/                   # utility scripts
├── tests/                     # test files
├── requirements.txt           # python dependencies
├── run_server.py             # server startup script
├── test_system.py            # system testing script
├── start.sh                  # quick start script
└── README.md                 # this file
```

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd /path/to/genetic_chatbot_backend
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. Start the server:
   ```bash
   python run_server.py
   ```

2. Test the system:
   ```bash
   python test_system.py
   ```

3. Access the API:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Chat Endpoint: http://localhost:8000/chat

## Configuration

The system uses YAML configuration with environment variable overrides. A default configuration file is created on first run.

### Configuration Options

- **VectorStore**: Collection name, embedding model, persistence directory
- **LLM**: Model selection, parameters, vLLM vs transformers
- **RAG**: Retrieval parameters, similarity thresholds
- **Server**: Host, port, logging, debug mode

### Environment Variables

You can override configuration using environment variables:

```bash
export VECTORSTORE_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export LLM_MODEL_NAME="microsoft/DialoGPT-medium"
export RAG_N_RESULTS=5
export SERVER_PORT=8000
```

## Data Sources

The system currently processes three CSV files:

1. **GCCG.csv**: Genetic Counseling Common Glossary with terms and definitions
2. **Genetic_bot_qb_with_genes_rerun_v3.csv**: Genetic counseling questions with gene information
3. **genetic_counsellor_training_questions_with_official_sources.csv**: Training questions for genetic counselors

### Adding New Data Sources

1. **CSV Files**: Place in project root, system auto-detects based on filename
2. **Other Formats**: Use the document ingestion API or place in `data/` directory
3. **API Ingestion**: Use the `/knowledge/ingest` endpoint

## API Endpoints

### Core Endpoints

- `POST /chat` - Main chat endpoint for genetic counseling questions
- `GET /health` - System health check
- `GET /knowledge/stats` - Knowledge base statistics

### Knowledge Management

- `POST /knowledge/ingest` - Ingest new documents
- `POST /knowledge/search` - Direct knowledge base search
- `GET /config` - Get current configuration

### Feedback

- `POST /feedback` - Submit user feedback on responses

## Usage Examples

### Basic Chat Query

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is gene therapy?"}'
```

### Filtered Search

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the inheritance patterns?",
    "filters": {"category": "inheritance_patterns"}
  }'
```

### Document Ingestion

```bash
curl -X POST "http://localhost:8000/knowledge/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "new_document.txt",
    "source": "New Source",
    "category": "educational"
  }'
```

## System Components

### Data Processor (`data_processor.py`)

Handles various data formats and creates searchable chunks:
- CSV processing with specialized handlers for different file types
- Text file processing with paragraph splitting
- JSON and YAML processing
- Metadata extraction and enrichment

### Enhanced Vectorstore (`enhanced_vectorstore.py`)

Advanced vector database operations:
- ChromaDB integration with persistence
- Semantic search with similarity scoring
- Metadata filtering and advanced queries
- Collection management and statistics

### Document Ingestion (`document_ingestion.py`)

Modular document ingestion system:
- Support for multiple file formats
- Automatic metadata detection
- Batch processing capabilities
- Ingestion logging and tracking

### Configuration Management (`config.py`)

Comprehensive configuration system:
- YAML-based configuration
- Environment variable overrides
- Runtime configuration updates
- Validation and defaults

## Testing

### Component Testing

```bash
python test_system.py
```

### API Testing

```bash
# Start server in one terminal
python run_server.py

# Test API in another terminal
python test_system.py --api
```

### Manual Testing

Use the interactive API documentation at http://localhost:8000/docs

## Customization

### Adding New File Processors

1. Add processor method to `DocumentIngestionManager`
2. Register in `file_processors` dictionary
3. Update supported formats in configuration

### Customizing Data Processing

1. Extend `DataProcessor` class
2. Add new processing methods
3. Update CSV file type detection

### Adding New LLM Models

1. Update `LLMConfig` in `config.py`
2. Modify LLM initialization in `main.py`
3. Add model-specific parameters

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Port Conflicts**: Change port in configuration or environment variables
3. **Memory Issues**: Reduce batch sizes or use smaller models
4. **File Not Found**: Check file paths and permissions

### Logging

The system provides comprehensive logging:
- Console output with configurable levels
- Optional file logging
- Structured log messages with timestamps

### Performance Optimization

- Use vLLM for better performance with large models
- Adjust chunk sizes for your data
- Configure similarity thresholds appropriately
- Use appropriate embedding models for your domain

## Development

### Adding New Features

1. Create new modules in the `app/` directory
2. Update imports in `main.py`
3. Add configuration options in `config.py`
4. Update tests in `test_system.py`

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Include error handling

## License

This project is for educational and research purposes. Please ensure compliance with any data usage agreements for the genetic counseling datasets.

## Support

For issues and questions:
1. Check the logs for error messages
2. Review the configuration settings
3. Test individual components using the test script
4. Consult the API documentation at `/docs`


<div align="center">
  <strong>Made with ❤️ by Sarvagna</strong>
</div> 



