"""
Document ingestion system for the genetic counseling RAG system.
Handles various document formats and provides easy integration for new sources.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import yaml
from datetime import datetime

from .enhanced_vectorstore import get_vectorstore
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)

class DocumentIngestionManager:
    """Manages document ingestion for the genetic counseling knowledge base."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 config_file: str = "ingestion_config.yaml"):
        """
        Initialize the document ingestion manager.
        
        Args:
            data_dir: Directory for storing data files
            config_file: Configuration file for ingestion settings
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.config_file = self.data_dir / config_file
        self.config = self._load_config()
        
        self.data_processor = DataProcessor(str(self.data_dir))
        self.vectorstore = get_vectorstore()
        
        # Supported file types and their processors
        self.file_processors = {
            '.csv': self._process_csv_file,
            '.txt': self._process_text_file,
            '.md': self._process_markdown_file,
            '.json': self._process_json_file,
            '.yaml': self._process_yaml_file,
            '.yml': self._process_yaml_file,
        }
        
        logger.info("Document ingestion manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load ingestion configuration from file."""
        default_config = {
            "auto_process_new_files": True,
            "supported_formats": [".csv", ".txt", ".md", ".json", ".yaml", ".yml"],
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "default_metadata": {
                "source": "unknown",
                "category": "general",
                "type": "document"
            },
            "csv_configs": {
                "gccg": {
                    "type": "glossary",
                    "category": "glossary",
                    "source": "GCCG Glossary"
                },
                "questions": {
                    "type": "qa",
                    "category": "qa",
                    "source": "Genetic Counseling Questions"
                },
                "training_questions": {
                    "type": "training",
                    "category": "training",
                    "source": "Genetic Counselor Training"
                }
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")
        else:
            # Create default config file
            self._save_config(default_config)
            logger.info(f"Created default configuration at {self.config_file}")
        
        return default_config
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _process_csv_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a CSV file based on its name and configuration."""
        try:
            file_name = file_path.stem.lower()
            
            # Check for known CSV types
            if 'gccg' in file_name:
                return self.data_processor.process_gccg_csv(str(file_path))
            elif 'question' in file_name and 'training' in file_name:
                return self.data_processor.process_training_questions_csv(str(file_path))
            elif 'question' in file_name:
                return self.data_processor.process_questions_csv(str(file_path))
            else:
                # Generic CSV processing
                return self._process_generic_csv(file_path, metadata)
                
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            return []
    
    def _process_generic_csv(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a generic CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            chunks = []
            
            # Create chunks from each row
            for idx, row in df.iterrows():
                # Combine all non-null values into text
                text_parts = []
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        text_parts.append(f"{col}: {value}")
                
                if text_parts:
                    chunk_text = "\n".join(text_parts)
                    chunks.append({
                        'text': chunk_text,
                        'source': metadata.get('source', file_path.name),
                        'category': metadata.get('category', 'csv_data'),
                        'type': 'csv_row',
                        'row_index': idx,
                        'file_name': file_path.name
                    })
            
            logger.info(f"Processed generic CSV {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing generic CSV {file_path}: {e}")
            return []
    
    def _process_text_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a plain text file."""
        return self.data_processor.process_text_file(
            str(file_path), 
            metadata.get('source', file_path.name),
            metadata.get('category', 'text')
        )
    
    def _process_markdown_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by headers and paragraphs
            sections = content.split('\n# ')
            chunks = []
            
            for i, section in enumerate(sections):
                if section.strip():
                    # Add # back to headers (except first section)
                    if i > 0:
                        section = '# ' + section
                    
                    # Clean up the section
                    section = section.strip()
                    if len(section) > 50:  # Only include substantial sections
                        chunks.append({
                            'text': section,
                            'source': metadata.get('source', file_path.name),
                            'category': metadata.get('category', 'markdown'),
                            'type': 'markdown_section',
                            'section_index': i,
                            'file_name': file_path.name
                        })
            
            logger.info(f"Processed Markdown file {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Markdown file {file_path}: {e}")
            return []
    
    def _process_json_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a JSON file."""
        return self.data_processor.process_json_file(
            str(file_path),
            metadata.get('source', file_path.name)
        )
    
    def _process_yaml_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            chunks = []
            
            # Convert YAML to text chunks
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (str, int, float)) and str(value).strip():
                        chunks.append({
                            'text': f"{key}: {value}",
                            'source': metadata.get('source', file_path.name),
                            'category': metadata.get('category', 'yaml_data'),
                            'type': 'yaml_field',
                            'field_name': key,
                            'file_name': file_path.name
                        })
                    elif isinstance(value, (list, dict)):
                        chunks.append({
                            'text': f"{key}: {json.dumps(value, indent=2)}",
                            'source': metadata.get('source', file_path.name),
                            'category': metadata.get('category', 'yaml_data'),
                            'type': 'yaml_complex',
                            'field_name': key,
                            'file_name': file_path.name
                        })
            
            logger.info(f"Processed YAML file {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing YAML file {file_path}: {e}")
            return []
    
    def ingest_file(self, 
                   file_path: Union[str, Path], 
                   metadata: Optional[Dict[str, Any]] = None,
                   auto_detect_metadata: bool = True) -> bool:
        """
        Ingest a single file into the knowledge base.
        
        Args:
            file_path: Path to the file to ingest
            metadata: Optional metadata for the file
            auto_detect_metadata: Whether to auto-detect metadata from filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Check if file type is supported
            if file_path.suffix.lower() not in self.config["supported_formats"]:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return False
            
            # Prepare metadata
            if metadata is None:
                metadata = self.config["default_metadata"].copy()
            
            if auto_detect_metadata:
                # Auto-detect metadata from filename
                file_name = file_path.stem.lower()
                
                # Check CSV configurations
                if file_path.suffix.lower() == '.csv':
                    for config_name, config_data in self.config["csv_configs"].items():
                        if config_name in file_name:
                            metadata.update(config_data)
                            break
                
                # Set source if not provided
                if metadata.get('source') == 'unknown':
                    metadata['source'] = file_path.name
                
                # Set category based on file type
                if metadata.get('category') == 'general':
                    if file_path.suffix.lower() == '.csv':
                        metadata['category'] = 'csv_data'
                    elif file_path.suffix.lower() == '.md':
                        metadata['category'] = 'markdown'
                    elif file_path.suffix.lower() in ['.yaml', '.yml']:
                        metadata['category'] = 'yaml_data'
            
            # Process the file
            processor = self.file_processors.get(file_path.suffix.lower())
            if not processor:
                logger.error(f"No processor found for file type: {file_path.suffix}")
                return False
            
            chunks = processor(file_path, metadata)
            
            if not chunks:
                logger.warning(f"No chunks generated from file: {file_path}")
                return False
            
            # Add chunks to vectorstore
            success = self.vectorstore.add_chunks(chunks)
            
            if success:
                logger.info(f"Successfully ingested {file_path} with {len(chunks)} chunks")
                # Log ingestion in metadata
                self._log_ingestion(file_path, len(chunks), metadata)
            
            return success
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return False
    
    def ingest_directory(self, 
                        directory_path: Union[str, Path],
                        recursive: bool = True,
                        file_pattern: Optional[str] = None) -> Dict[str, bool]:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            file_pattern: Optional glob pattern for file matching
            
        Returns:
            Dictionary mapping file paths to success status
        """
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists() or not directory_path.is_dir():
                logger.error(f"Directory not found: {directory_path}")
                return {}
            
            # Find files
            if file_pattern:
                files = list(directory_path.rglob(file_pattern) if recursive else directory_path.glob(file_pattern))
            else:
                files = []
                for ext in self.config["supported_formats"]:
                    if recursive:
                        files.extend(directory_path.rglob(f"*{ext}"))
                    else:
                        files.extend(directory_path.glob(f"*{ext}"))
            
            # Remove duplicates and filter existing files
            files = list(set(files))
            files = [f for f in files if f.is_file()]
            
            logger.info(f"Found {len(files)} files to ingest in {directory_path}")
            
            # Ingest each file
            results = {}
            for file_path in files:
                results[str(file_path)] = self.ingest_file(file_path)
            
            successful = sum(1 for success in results.values() if success)
            logger.info(f"Ingestion complete: {successful}/{len(files)} files successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {e}")
            return {}
    
    def _log_ingestion(self, file_path: Path, chunk_count: int, metadata: Dict[str, Any]) -> None:
        """Log ingestion details for tracking."""
        try:
            log_file = self.data_dir / "ingestion_log.json"
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "file_path": str(file_path),
                "chunk_count": chunk_count,
                "metadata": metadata
            }
            
            # Load existing log
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Add new entry
            log_data.append(log_entry)
            
            # Save updated log
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error logging ingestion: {e}")
    
    def get_ingestion_log(self) -> List[Dict[str, Any]]:
        """Get the ingestion log."""
        try:
            log_file = self.data_dir / "ingestion_log.json"
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error reading ingestion log: {e}")
            return []
    
    def reingest_all(self, csv_files: Optional[Dict[str, str]] = None) -> bool:
        """
        Reingest all data, clearing existing data first.
        
        Args:
            csv_files: Optional CSV files to reingest
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting reingestion of all data")
            
            # Clear existing collection (this is a limitation - ChromaDB doesn't have easy clear)
            # For now, we'll create a new collection
            from .enhanced_vectorstore import EnhancedVectorStore
            global _vectorstore_instance
            _vectorstore_instance = EnhancedVectorStore(
                collection_name=f"genetic_counseling_kb_{datetime.now().timestamp()}"
            )
            
            # Reingest CSV files if provided
            if csv_files:
                success = self.vectorstore.initialize_from_csv_files(csv_files)
                if not success:
                    logger.error("Failed to reingest CSV files")
                    return False
            
            logger.info("Reingestion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during reingestion: {e}")
            return False

# Global instance for easy access
_ingestion_manager = None

def get_ingestion_manager() -> DocumentIngestionManager:
    """Get the global ingestion manager instance."""
    global _ingestion_manager
    if _ingestion_manager is None:
        _ingestion_manager = DocumentIngestionManager()
    return _ingestion_manager
