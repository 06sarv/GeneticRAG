"""
Data processing module for genetic counseling RAG system.
Handles CSV files and other document sources for knowledge base ingestion.
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes various data sources for the genetic counseling knowledge base."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def process_gccg_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process the GCCG (Genetic Counseling Common Glossary) CSV file."""
        try:
            df = pd.read_csv(file_path)
            chunks = []
            
            for _, row in df.iterrows():
                # Create comprehensive text chunks from each row
                term = row.get('Term', '')
                abbreviation = row.get('Abbreviation', '')
                category = row.get('Category', '')
                plain_def = row.get('Plain-language definition', '')
                formal_def = row.get('Formal/technical definition', '')
                example = row.get('Example/Notes', '')
                sources = row.get('Primary sources/references', '')
                url = row.get('Source URL', '')
                
                # Create multiple text variations for better retrieval
                if term and plain_def:
                    # Main definition chunk
                    main_text = f"Term: {term}"
                    if abbreviation:
                        main_text += f" ({abbreviation})"
                    main_text += f"\nCategory: {category}\nDefinition: {plain_def}"
                    if formal_def:
                        main_text += f"\nTechnical definition: {formal_def}"
                    if example:
                        main_text += f"\nExample: {example}"
                    if sources:
                        main_text += f"\nSources: {sources}"
                    
                    chunks.append({
                        'text': main_text,
                        'source': 'GCCG Glossary',
                        'category': 'glossary',
                        'term': term,
                        'abbreviation': abbreviation,
                        'url': url,
                        'type': 'definition'
                    })
                    
                    # Create searchable variations
                    if abbreviation and abbreviation != term:
                        chunks.append({
                            'text': f"Abbreviation: {abbreviation} stands for {term}. {plain_def}",
                            'source': 'GCCG Glossary',
                            'category': 'glossary',
                            'term': term,
                            'abbreviation': abbreviation,
                            'url': url,
                            'type': 'abbreviation'
                        })
                    
                    # Create category-specific chunk
                    if category:
                        chunks.append({
                            'text': f"In {category}: {term} - {plain_def}",
                            'source': 'GCCG Glossary',
                            'category': f'glossary_{category.lower().replace(" ", "_")}',
                            'term': term,
                            'abbreviation': abbreviation,
                            'url': url,
                            'type': 'category'
                        })
            
            logger.info(f"Processed GCCG CSV: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing GCCG CSV: {e}")
            return []
    
    def process_questions_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process the genetic counseling questions CSV file."""
        try:
            df = pd.read_csv(file_path)
            chunks = []
            
            for _, row in df.iterrows():
                question = row.get('Question', '')
                main_problem = row.get('Main problem in the question', '')
                problem_type = row.get('Type of problem', '')
                source = row.get('Source', '')
                source_url = row.get('Source URL', '')
                gene_affected = row.get('Gene affected', '')
                gene_source_url = row.get('Gene Source URL', '')
                misconception = row.get('Common misconception', '')
                misconception_source = row.get('Misconception Source URL', '')
                
                if question:
                    # Main Q&A chunk
                    qa_text = f"Question: {question}"
                    if main_problem:
                        qa_text += f"\nMain problem: {main_problem}"
                    if problem_type:
                        qa_text += f"\nType: {problem_type}"
                    if gene_affected:
                        qa_text += f"\nGene(s) involved: {gene_affected}"
                    if source:
                        qa_text += f"\nSource: {source}"
                    
                    chunks.append({
                        'text': qa_text,
                        'source': source or 'Genetic Counseling Questions',
                        'category': 'qa',
                        'problem_type': problem_type,
                        'gene_affected': gene_affected,
                        'source_url': source_url,
                        'gene_source_url': gene_source_url,
                        'type': 'question'
                    })
                    
                    # Create gene-specific chunks if genes are mentioned
                    if gene_affected and pd.notna(gene_affected):
                        gene_list = [g.strip() for g in str(gene_affected).split(';') if g.strip()]
                        for gene in gene_list:
                            chunks.append({
                                'text': f"Gene: {gene}\nRelated question: {question}\nProblem: {main_problem}",
                                'source': source or 'Genetic Counseling Questions',
                                'category': 'gene_specific',
                                'gene': gene,
                                'source_url': source_url,
                                'type': 'gene_question'
                            })
                    
                    # Create misconception chunk if available
                    if misconception:
                        chunks.append({
                            'text': f"Common misconception about {main_problem}: {misconception}\nCorrect information: {question}",
                            'source': source or 'Genetic Counseling Questions',
                            'category': 'misconception',
                            'problem_type': problem_type,
                            'misconception_source': misconception_source,
                            'type': 'misconception'
                        })
            
            logger.info(f"Processed Questions CSV: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Questions CSV: {e}")
            return []
    
    def process_training_questions_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process the genetic counselor training questions CSV file."""
        try:
            df = pd.read_csv(file_path)
            chunks = []
            
            for _, row in df.iterrows():
                question = row.get('Question', '')
                main_problem = row.get('Main problem in the question', '')
                problem_type = row.get('Type of problem', '')
                source = row.get('Source', '')
                source_url = row.get('Source URL', '')
                
                if question:
                    # Main training question chunk
                    training_text = f"Training Question: {question}"
                    if main_problem:
                        training_text += f"\nMain problem: {main_problem}"
                    if problem_type:
                        training_text += f"\nProblem type: {problem_type}"
                    if source:
                        training_text += f"\nSource: {source}"
                    
                    chunks.append({
                        'text': training_text,
                        'source': source or 'Genetic Counselor Training',
                        'category': 'training',
                        'problem_type': problem_type,
                        'source_url': source_url,
                        'type': 'training_question'
                    })
                    
                    # Create problem-type specific chunks
                    if problem_type:
                        chunks.append({
                            'text': f"Problem type: {problem_type}\nExample question: {question}\nMain issue: {main_problem}",
                            'source': source or 'Genetic Counselor Training',
                            'category': f'training_{problem_type.lower().replace(" ", "_")}',
                            'problem_type': problem_type,
                            'source_url': source_url,
                            'type': 'problem_type'
                        })
            
            logger.info(f"Processed Training Questions CSV: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Training Questions CSV: {e}")
            return []
    
    def process_text_file(self, file_path: str, source_name: str, category: str = "general") -> List[Dict[str, Any]]:
        """Process a plain text file by splitting into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into paragraphs or sections
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            chunks = []
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # Only include substantial paragraphs
                    chunks.append({
                        'text': paragraph,
                        'source': source_name,
                        'category': category,
                        'type': 'text_chunk',
                        'chunk_id': i
                    })
            
            logger.info(f"Processed text file {file_path}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def process_json_file(self, file_path: str, source_name: str) -> List[Dict[str, Any]]:
        """Process a JSON file containing structured data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Create text from dictionary values
                        text_parts = []
                        for key, value in item.items():
                            if isinstance(value, str) and value.strip():
                                text_parts.append(f"{key}: {value}")
                        
                        if text_parts:
                            chunks.append({
                                'text': '\n'.join(text_parts),
                                'source': source_name,
                                'category': 'json_data',
                                'type': 'json_chunk',
                                'chunk_id': i
                            })
            
            logger.info(f"Processed JSON file {file_path}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")
            return []
    
    def process_all_csv_files(self, csv_files: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process all specified CSV files."""
        all_chunks = []
        
        for file_type, file_path in csv_files.items():
            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            if file_type == 'gccg':
                chunks = self.process_gccg_csv(file_path)
            elif file_type == 'questions':
                chunks = self.process_questions_csv(file_path)
            elif file_type == 'training_questions':
                chunks = self.process_training_questions_csv(file_path)
            else:
                logger.warning(f"Unknown file type: {file_type}")
                continue
            
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
    
    def save_processed_data(self, chunks: List[Dict[str, Any]], output_file: str = "processed_data.json"):
        """Save processed chunks to a JSON file for backup."""
        output_path = self.data_dir / output_file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def load_processed_data(self, input_file: str = "processed_data.json") -> List[Dict[str, Any]]:
        """Load previously processed data from JSON file."""
        input_path = self.data_dir / input_file
        try:
            if input_path.exists():
                with open(input_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
                return chunks
            else:
                logger.info(f"No processed data file found at {input_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return []
