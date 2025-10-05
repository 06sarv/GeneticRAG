"""
Configuration management for the genetic counseling RAG system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Configuration for the vectorstore."""
    collection_name: str = "genetic_counseling_kb"
    embedding_model: str = "all-MiniLM-L6-v2"
    persist_directory: str = "data/chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class LLMConfig:
    """Configuration for the language model."""
    model_name: str = "microsoft/DialoGPT-medium"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_vllm: bool = False # Disabling vLLM to troubleshoot safetensors issue
    fallback_to_simple: bool = True
    max_model_len: int = 1024 # Adjusted to match model's max_position_embeddings

@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    n_results: int = 5
    min_similarity: float = 0.3
    max_context_length: int = 4000
    include_metadata: bool = True
    rerank_results: bool = False

@dataclass
class ServerConfig:
    """Configuration for the FastAPI server."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"

@dataclass
class AppConfig:
    """Main application configuration."""
    vectorstore: VectorStoreConfig
    llm: LLMConfig
    rag: RAGConfig
    server: ServerConfig
    data_dir: str = "data"
    log_file: Optional[str] = None
    enable_feedback: bool = True
    enable_analytics: bool = False

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Create config objects from data
                vectorstore_config = VectorStoreConfig(**config_data.get('vectorstore', {}))
                llm_config = LLMConfig(**config_data.get('llm', {}))
                rag_config = RAGConfig(**config_data.get('rag', {}))
                server_config = ServerConfig(**config_data.get('server', {}))
                
                app_config = AppConfig(
                    vectorstore=vectorstore_config,
                    llm=llm_config,
                    rag=rag_config,
                    server=server_config,
                    **config_data.get('app', {})
                )
                
                logger.info(f"Loaded configuration from {self.config_file}")
                return app_config
                
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")
                return self._create_default_config()
        else:
            logger.info("No config file found. Creating default configuration.")
            config = self._create_default_config()
            self.save_config(config)
            return config
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration."""
        return AppConfig(
            vectorstore=VectorStoreConfig(),
            llm=LLMConfig(),
            rag=RAGConfig(),
            server=ServerConfig()
        )
    
    def save_config(self, config: Optional[AppConfig] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save. If None, saves current config.
        """
        if config is None:
            config = self.config
        
        try:
            config_data = {
                'vectorstore': asdict(config.vectorstore),
                'llm': asdict(config.llm),
                'rag': asdict(config.rag),
                'server': asdict(config.server),
                'app': {
                    'data_dir': config.data_dir,
                    'log_file': config.log_file,
                    'enable_feedback': config.enable_feedback,
                    'enable_analytics': config.enable_analytics
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration values to update
        """
        try:
            # Update vectorstore config
            if 'vectorstore' in kwargs:
                for key, value in kwargs['vectorstore'].items():
                    if hasattr(self.config.vectorstore, key):
                        setattr(self.config.vectorstore, key, value)
            
            # Update LLM config
            if 'llm' in kwargs:
                for key, value in kwargs['llm'].items():
                    if hasattr(self.config.llm, key):
                        setattr(self.config.llm, key, value)
            
            # Update RAG config
            if 'rag' in kwargs:
                for key, value in kwargs['rag'].items():
                    if hasattr(self.config.rag, key):
                        setattr(self.config.rag, key, value)
            
            # Update server config
            if 'server' in kwargs:
                for key, value in kwargs['server'].items():
                    if hasattr(self.config.server, key):
                        setattr(self.config.server, key, value)
            
            # Update app config
            for key, value in kwargs.items():
                if key not in ['vectorstore', 'llm', 'rag', 'server'] and hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logger.info("Configuration updated")
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
    
    def get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # VectorStore overrides
        if os.getenv('VECTORSTORE_COLLECTION_NAME'):
            overrides.setdefault('vectorstore', {})['collection_name'] = os.getenv('VECTORSTORE_COLLECTION_NAME')
        if os.getenv('VECTORSTORE_EMBEDDING_MODEL'):
            overrides.setdefault('vectorstore', {})['embedding_model'] = os.getenv('VECTORSTORE_EMBEDDING_MODEL')
        if os.getenv('VECTORSTORE_PERSIST_DIRECTORY'):
            overrides.setdefault('vectorstore', {})['persist_directory'] = os.getenv('VECTORSTORE_PERSIST_DIRECTORY')
        
        # LLM overrides
        if os.getenv('LLM_MODEL_NAME'):
            overrides.setdefault('llm', {})['model_name'] = os.getenv('LLM_MODEL_NAME')
        if os.getenv('LLM_MAX_TOKENS'):
            overrides.setdefault('llm', {})['max_tokens'] = int(os.getenv('LLM_MAX_TOKENS'))
        if os.getenv('LLM_TEMPERATURE'):
            overrides.setdefault('llm', {})['temperature'] = float(os.getenv('LLM_TEMPERATURE'))
        if os.getenv('LLM_USE_VLLM'):
            overrides.setdefault('llm', {})['use_vllm'] = os.getenv('LLM_USE_VLLM').lower() == 'true'
        
        # RAG overrides
        if os.getenv('RAG_N_RESULTS'):
            overrides.setdefault('rag', {})['n_results'] = int(os.getenv('RAG_N_RESULTS'))
        if os.getenv('RAG_MIN_SIMILARITY'):
            overrides.setdefault('rag', {})['min_similarity'] = float(os.getenv('RAG_MIN_SIMILARITY'))
        
        # Server overrides
        if os.getenv('SERVER_HOST'):
            overrides.setdefault('server', {})['host'] = os.getenv('SERVER_HOST')
        if os.getenv('SERVER_PORT'):
            overrides.setdefault('server', {})['port'] = int(os.getenv('SERVER_PORT'))
        if os.getenv('SERVER_DEBUG'):
            overrides.setdefault('server', {})['debug'] = os.getenv('SERVER_DEBUG').lower() == 'true'
        if os.getenv('LOG_LEVEL'):
            overrides.setdefault('server', {})['log_level'] = os.getenv('LOG_LEVEL')
        
        # App overrides
        if os.getenv('DATA_DIR'):
            overrides['data_dir'] = os.getenv('DATA_DIR')
        if os.getenv('LOG_FILE'):
            overrides['log_file'] = os.getenv('LOG_FILE')
        if os.getenv('ENABLE_FEEDBACK'):
            overrides['enable_feedback'] = os.getenv('ENABLE_FEEDBACK').lower() == 'true'
        if os.getenv('ENABLE_ANALYTICS'):
            overrides['enable_analytics'] = os.getenv('ENABLE_ANALYTICS').lower() == 'true'
        
        return overrides
    
    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides to the configuration."""
        overrides = self.get_env_overrides()
        if overrides:
            self.update_config(**overrides)
            logger.info("Applied environment variable overrides")

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.apply_env_overrides()
    return _config_manager

def get_config() -> AppConfig:
    """Get the current application configuration."""
    return get_config_manager().get_config()
