#!/usr/bin/env python3
"""
Startup script for the GeneticRAG system.

This script initializes the system and starts the FastAPI server.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """main function to start the server"""
    try:
        # import and run the fastapi app
        from app.main import app
        import uvicorn
        from app.config import get_config
        
        # get configuration
        config = get_config()
        
        # setup logging
        logging.basicConfig(
            level=getattr(logging, config.server.log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Starting GeneticRAG system...")
        
        # print configuration
        logger.info(f"Server: {config.server.host}:{config.server.port}")
        logger.info(f"Debug mode: {config.server.debug}")
        logger.info(f"Vectorstore: {config.vectorstore.collection_name}")
        logger.info(f"LLM: {config.llm.model_name} (vLLM: {config.llm.use_vllm})")
        
        # start the server
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            reload=config.server.reload,
            log_level=config.server.log_level.lower()
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
