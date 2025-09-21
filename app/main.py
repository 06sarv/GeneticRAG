"""
Main FastAPI application for the genetic counseling RAG system.

This module provides a REST API for querying genetic counseling information
using a retrieval-augmented generation (RAG) approach with vector search.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import our modules
from .config import get_config, AppConfig
from .enhanced_vectorstore import get_vectorstore, initialize_vectorstore
from .document_ingestion import get_ingestion_manager
from .data_processor import DataProcessor
from .vcf_processor import VCFProcessor

# Load environment variables
load_dotenv()

def setup_logging(config: AppConfig):
    """Setup logging configuration."""
    log_level = getattr(logging, config.server.log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# Initialize configuration
config = get_config()
setup_logging(config)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Genetic Counseling RAG System",
    description="A RAG-based system for answering genetic counseling questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question or message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    filters: Optional[Dict[str, str]] = Field(None, description="Search filters")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Bot's response")
    sources: List[str] = Field(..., description="Sources used for the response")
    confidence: Optional[float] = Field(None, description="Confidence score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FeedbackRequest(BaseModel):
    user_message: str = Field(..., description="Original user message")
    bot_reply: str = Field(..., description="Bot's reply")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comments: Optional[str] = Field(None, description="Additional comments")

class DocumentIngestionRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to ingest")
    source: Optional[str] = Field(None, description="Source name for the document")
    category: Optional[str] = Field(None, description="Category for the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# Global variables for lazy initialization
_vectorstore = None
_ingestion_manager = None
_llm = None
_vcf_processor = None

def get_vectorstore_instance():
    """Get or initialize the vectorstore instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore

def get_ingestion_manager_instance():
    """Get or initialize the ingestion manager instance."""
    global _ingestion_manager
    if _ingestion_manager is None:
        _ingestion_manager = get_ingestion_manager()
    return _ingestion_manager

def get_vcf_processor_instance():
    global _vcf_processor
    if _vcf_processor is None:
        _vcf_processor = VCFProcessor()
        logger.info("Initialized VCF processor")
    return _vcf_processor

def get_llm_instance():
    """Get or initialize the LLM instance."""
    global _llm
    if _llm is None:
        try:
            if config.llm.use_vllm:
                from vllm import LLM

                _llm = LLM(
                    model=config.llm.model_name,
                    trust_remote_code=True,
                    max_model_len=config.llm.max_model_len,
                )
                logger.info(f"Initialized vLLM with model: {config.llm.model_name}")
            else:
                from transformers import pipeline

                _llm = pipeline(
                    "text-generation",
                    model=config.llm.model_name,
                    trust_remote_code=True,
                )
                logger.info(f"Initialized transformers pipeline with model: {config.llm.model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            if config.llm.fallback_to_simple:
                _llm = None
                logger.warning("LLM initialization failed, using simple fallback")
            # No else here, _llm is already None or will be returned
    return _llm


# Utility functions
def is_variant_query(query: str) -> bool:

    """check if the query is about a specific genetic variant"""
    query_lower = query.lower()
    
    # hgvs notation patterns
    hgvs_patterns = [
        r'[cpg]\.\d+[+-]?\d*[ATCG]>[ATCG]',  # c.123A>G, p.456L>P, g.789C>T
        r'[cpg]\.\d+[+-]?\d*[ATCG]',  # c.123A, p.456L, g.789C
        r'rs\d+',  # rs123456
        r'CA\d+',  # CA123456789
        r'VCV\d+',  # VCV000012345
    ]
    
    for pattern in hgvs_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # gene symbol patterns
    gene_pattern = r'\b[A-Z]{2,6}[0-9]*\b'
    genes = re.findall(gene_pattern, query)
    common_words = {'AND', 'THE', 'FOR', 'ARE', 'CAN', 'MAY', 'NOT', 'ALL', 'ANY', 'DNA', 'RNA', 'PCR', 'CVS', 'PGT', 'WES', 'WGS'}
    genes = [g for g in genes if g not in common_words]
    
    if genes:
        return True
    
    # variant-related keywords
    variant_keywords = [
        'variant', 'mutation', 'pathogenic', 'benign', 'vus', 'variant of uncertain significance',
        'missense', 'nonsense', 'frameshift', 'splice', 'deletion', 'duplication',
        'cnv', 'copy number variant', 'structural variant', 'sv'
    ]
    
    if any(keyword in query_lower for keyword in variant_keywords):
        return True
    
    return False

def variant_query_handler(query: str) -> str:
    """handle variant-specific queries"""
    # this is a placeholder for variant analysis pipeline
    return f"""I understand you're asking about a genetic variant. This query has been identified as variant-specific and would be processed by our specialized variant analysis pipeline.

Query: "{query}"

For detailed variant interpretation, please consult with a qualified genetic counselor or clinical geneticist who can provide personalized analysis based on:
- The specific variant and its classification
- Your family history
- Clinical context
- Current literature and databases

This system is designed for general genetic education and counseling questions. For variant-specific analysis, professional interpretation is recommended."""

def build_prompt(user_query: str, retrieved_chunks: List[Dict[str, Any]], config: AppConfig) -> str:
    """Build a comprehensive prompt for the LLM."""
    # Prepare context from retrieved chunks
    context_parts = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_part = f"{idx}. {chunk['text']}"
        if chunk.get('source'):
            context_part += f" [Source: {chunk['source']}]"
        if chunk.get('category'):
            context_part += f" [Category: {chunk['category']}]"
        context_parts.append(context_part)
    
    context = "\n\n".join(context_parts)
    
    # Build the prompt
    prompt = f"""You are a knowledgeable genetic counseling assistant with expertise in genetics, genomics, and genetic counseling. You provide accurate, evidence-based information while always encouraging consultation with qualified healthcare professionals for medical decisions.

CONTEXT INFORMATION:
{context}

USER QUESTION: {user_query}

INSTRUCTIONS:
1. Provide a clear, accurate, and helpful answer based on the context information provided
2. Cite the source numbers (e.g., [1], [2]) when referencing specific information
3. If the question is about a specific genetic variant, recommend consulting with a qualified genetic counselor
4. If the question is general, provide educational information while encouraging professional consultation for medical decisions
5. If you cannot find relevant information in the context, say so and suggest consulting a genetic counselor
6. Always emphasize that this information is for educational purposes and not a substitute for professional medical advice
7. Be empathetic and supportive in your responses
8. Keep responses concise but comprehensive

RESPONSE:"""
    
    return prompt

def generate_response(user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Generate a response using the LLM or fallback method."""
    try:
        llm = get_llm_instance()
        
        if llm is None:
            # Fallback response when LLM is not available
            return generate_fallback_response(user_query, retrieved_chunks)
        
        # Build prompt
        prompt = build_prompt(user_query, retrieved_chunks, config)
        
        # Generate response
        if config.llm.use_vllm:
            # vLLM generation
            response = llm.generate(prompt)
            return response.completions[0].text.strip()
        else:
            # Transformers pipeline generation
            response = llm(prompt, max_length=config.llm.max_tokens, do_sample=True)
            return response[0]['generated_text'].replace(prompt, '').strip()
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return generate_fallback_response(user_query, retrieved_chunks)

def generate_fallback_response(user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Generate a fallback response when LLM is not available."""
    if not retrieved_chunks:
        return """I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing your question or consult with a qualified genetic counselor for personalized assistance.

This information is for educational purposes only and is not a substitute for professional medical advice."""
    
    response = "Based on the available information:\n\n"
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        response += f"{i}. {chunk['text']}"
        if chunk.get('source'):
            response += f" [Source: {chunk['source']}]"
        response += "\n\n"
    
    response += """Note: This is a basic response based on available information. For more detailed analysis and personalized guidance, please consult with a qualified genetic counselor.

This information is for educational purposes only and is not a substitute for professional medical advice."""
    
    return response

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Genetic Counseling RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        vectorstore = get_vectorstore_instance()
        stats = vectorstore.get_collection_stats()
        
        return {
            "status": "healthy",
            "service": "genetic_chatbot_backend",
            "vectorstore": {
                "status": "connected",
                "documents": stats.get("total_documents", 0)
            },
            "llm": {
                "status": "available" if get_llm_instance() is not None else "fallback"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for genetic counseling questions."""
    try:
        user_query = request.message.strip()
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"Received query: {user_query[:100]}...")
        
        # Check if this is a variant query
        if is_variant_query(user_query):
            reply = variant_query_handler(user_query)
            sources = ["Variant Analysis Pipeline"]
            confidence = 0.8  # High confidence for variant routing
        else:
            # Use RAG system for general genetics questions
            vectorstore = get_vectorstore_instance()
            
            # Apply filters if provided
            filters = request.filters or {}
            category_filter = filters.get('category')
            source_filter = filters.get('source')
            type_filter = filters.get('type')
            
            # Retrieve similar chunks
            retrieved_chunks = vectorstore.retrieve_similar_chunks(
                query=user_query,
                n_results=config.rag.n_results,
                category_filter=category_filter,
                source_filter=source_filter,
                type_filter=type_filter,
                min_similarity=config.rag.min_similarity
            )
            
            if retrieved_chunks:
                # Generate response
                reply = generate_response(user_query, retrieved_chunks)
                sources = [chunk['source'] for chunk in retrieved_chunks]
                confidence = retrieved_chunks[0].get('similarity_score', 0.5)
            else:
                reply = """I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing your question or consult with a qualified genetic counselor for personalized assistance.

This information is for educational purposes only and is not a substitute for professional medical advice."""
                sources = []
                confidence = 0.0
        
        return ChatResponse(
            reply=reply,
            sources=sources,
            confidence=confidence,
            metadata={
                "query_type": "variant" if is_variant_query(user_query) else "general",
                "chunks_retrieved": len(retrieved_chunks) if 'retrieved_chunks' in locals() else 0,
                "filters_applied": request.filters
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback")
async def feedback_endpoint(feedback: FeedbackRequest):
    """Collect user feedback on responses."""
    if not config.enable_feedback:
        raise HTTPException(status_code=404, detail="Feedback collection is disabled")
    
    try:
        # Log feedback (in a real system, you'd store this in a database)
        logger.info(f"Feedback received - Rating: {feedback.rating}, Comments: {feedback.comments}")
        
        # Here you could store feedback in a database or send to analytics
        # For now, we'll just log it
        
        return {"status": "success", "message": "Feedback received"}
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail="Error processing feedback")

@app.get("/knowledge/stats")
async def get_knowledge_stats():
    """Get statistics about the knowledge base."""
    try:
        vectorstore = get_vectorstore_instance()
        stats = vectorstore.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving knowledge statistics")

@app.post("/knowledge/ingest")
async def ingest_document(request: DocumentIngestionRequest):
    """Ingest a new document into the knowledge base."""
    try:
        ingestion_manager = get_ingestion_manager_instance()
        
        metadata = request.metadata or {}
        if request.source:
            metadata['source'] = request.source
        if request.category:
            metadata['category'] = request.category
        
        success = ingestion_manager.ingest_file(
            file_path=request.file_path,
            metadata=metadata
        )
        
        if success:
            return {"status": "success", "message": "Document ingested successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to ingest document")
            
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail="Error ingesting document")

# VCF Processing API Models
class VCFProcessRequest(BaseModel):
    """Request model for VCF processing."""
    file_path: str = Field(..., description="Path to the VCF file to process")
    patient_id: Optional[str] = Field(None, description="Patient ID (will be anonymized)")
    analysis_type: Literal["variant", "dbsnp", "prs", "pathway", "novel_recurrent", "functional_prediction", "acmg_interpretation", "network_pharmacology", "all"] = "all"
    output_dir: Optional[str] = None


class VCFProcessResponse(BaseModel):
    """Response model for VCF processing."""
    success: bool
    anonymized_id: str
    results: Dict[str, Any]
    
@app.post("/api/process_vcf", response_model=VCFProcessResponse)
async def process_vcf(request: VCFProcessRequest):
    """Process a VCF file and return analysis results."""
    try:
        vcf_processor = get_vcf_processor_instance()
        
        # Anonymize patient ID if provided
        anonymized_id = vcf_processor.anonymize_patient_id(request.patient_id) if request.patient_id else "anonymous"
        
        # Process the VCF file based on requested analysis type
        results = {}
        
        if request.analysis_type in ["variant", "all"]:
            variants = vcf_processor.parse_vcf(request.file_path, request.output_dir)
            variant_types = vcf_processor.determine_variant_types(variants)
            results["variants"] = {
                "count": len(variants),
                "types": variant_types
            }
            
        if request.analysis_type in ["dbsnp", "all"]:
            variants = vcf_processor.parse_vcf(request.file_path, request.output_dir) if "variants" not in results else variants
            dbsnp_results = vcf_processor.query_dbsnp(variants)
            results["dbsnp"] = dbsnp_results
            
        if request.analysis_type in ["prs", "all"]:
            variants = vcf_processor.parse_vcf(request.file_path, request.output_dir) if "variants" not in results else variants
            prs_score = vcf_processor.calculate_prs(variants)
            results["prs"] = prs_score
            
        if request.analysis_type in ["pathway", "all"]:
            variants = vcf_processor.parse_vcf(request.file_path, request.output_dir) if "variants" not in results else variants
            pathway_analysis = vcf_processor.analyze_pathways(variants)
            results["pathways"] = pathway_analysis
            
        if request.analysis_type in ["novel_recurrent", "all"]:
            variants = vcf_processor.parse_vcf(request.file_path, request.output_dir) if "variants" not in results else variants
            novel_recurrent_variants = vcf_processor.identify_novel_recurrent_variants(variants)
            results["novel_recurrent_variants"] = novel_recurrent_results
            
        if analysis_type == "functional_prediction" or analysis_type == "all":
            if not vcf_path:
                vcf_path = await vcf_processor.parse_vcf(vcf_content, output_dir=request.output_dir)
            # Assuming we want to predict functional impact for all variants in the VCF
            # For a real application, you might want to filter variants first
            variants_to_predict = vcf_processor.get_variants_from_vcf(vcf_path) # You'll need to implement this method in VCFProcessor
            functional_prediction_results = []
            for variant in variants_to_predict:
                prediction = vcf_processor.predict_functional_impact(variant)
                functional_prediction_results.append({"variant": variant, "prediction": prediction})
            results["functional_prediction"] = functional_prediction_results

        # Perform network pharmacology analysis
        if analysis_type == "network_pharmacology" or analysis_type == "all":
            if not variants:
                variants = vcf_processor.get_variants_from_vcf(vcf_file_path)
            
            network_pharmacology_results = []
            for variant in variants:
                network_pharmacology_results.append(vcf_processor.perform_network_pharmacology_analysis(variant))
            response.network_pharmacology_results = network_pharmacology_results

        # Perform ACMG interpretation
        if analysis_type == "acmg_interpretation" or analysis_type == "all":
            if not variants:
                variants = vcf_processor.get_variants_from_vcf(vcf_file_path)
            
            acmg_results = []
            for variant in variants:
                # For ACMG, we need functional predictions and gnomAD allele frequencies
                functional_predictions = vcf_processor.predict_functional_impact(variant)
                gnomad_allele_frequency = vcf_processor.query_gnomad(variant)
                acmg_results.append(vcf_processor.interpret_variant_acmg(variant, functional_predictions, gnomad_allele_frequency))
            response.acmg_interpretation_results = acmg_results

        if not results:
            return VCFProcessResponse(
                success=True,
                anonymized_id=anonymized_id,
                results=results
            )
    except Exception as e:
        logger.error(f"Error processing VCF file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/search")
async def search_knowledge(query: str, 
                          category: Optional[str] = None,
                          source: Optional[str] = None,
                          type: Optional[str] = None,
                          limit: int = 10):
    """Search the knowledge base directly."""
    try:
        vectorstore = get_vectorstore_instance()
        
        results = vectorstore.retrieve_similar_chunks(
            query=query,
            n_results=limit,
            category_filter=category,
            source_filter=source,
            type_filter=type
        )
        
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail="Error searching knowledge base")

@app.get("/config")
async def get_configuration():
    """Get current configuration (read-only)."""
    return {
        "vectorstore": {
            "collection_name": config.vectorstore.collection_name,
            "embedding_model": config.vectorstore.embedding_model,
            "persist_directory": config.vectorstore.persist_directory
        },
        "llm": {
            "model_name": config.llm.model_name,
            "use_vllm": config.llm.use_vllm,
            "max_tokens": config.llm.max_tokens
        },
        "rag": {
            "n_results": config.rag.n_results,
            "min_similarity": config.rag.min_similarity
        },
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "debug": config.server.debug
        }
    }

# Initialize the system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    try:
        logger.info("Starting Genetic Counseling RAG System...")
        
        # Initialize vectorstore with CSV files
        csv_files = {
            'gccg': 'data/raw/GCCG.csv',
            'questions': 'data/raw/Genetic_bot_qb_with_genes_rerun_v3.csv',
            'training_questions': 'data/raw/genetic_counsellor_training_questions_with_official_sources.csv'
        }
        
        # Check if CSV files exist
        existing_csv_files = {}
        for key, file_path in csv_files.items():
            if os.path.exists(file_path):
                existing_csv_files[key] = file_path
            else:
                logger.warning(f"CSV file not found: {file_path}")
        
        if existing_csv_files:
            # Initialize vectorstore
            vectorstore = get_vectorstore_instance()
            success = vectorstore.initialize_from_csv_files(existing_csv_files)
            
            if success:
                logger.info("Vectorstore initialized successfully with CSV data")
            else:
                logger.error("Failed to initialize vectorstore with CSV data")
        else:
            logger.warning("No CSV files found, vectorstore will be empty")
        
        # Initialize LLM
        try:
            get_llm_instance()
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
        
        logger.info("System startup completed")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level.lower()
    )