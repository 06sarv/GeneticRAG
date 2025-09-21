"""
VCF file processing module for genetic counseling RAG system.
Handles Variant Call Format (VCF) files for genetic variant analysis while maintaining patient confidentiality.
"""

import os
import logging
import pandas as pd
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re

# For VCF parsing
try:
    import pysam
except ImportError:
    logging.warning("pysam not installed. Install with 'pip install pysam' for VCF file processing.")

# For API calls to external databases
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class VCFProcessor:
    """Processes VCF files for genetic variant analysis with privacy protection."""
    
    def __init__(self, data_dir: str = "data/vcf", temp_dir: str = "data/temp"):
        """
        Initialize the VCF processor.
        
        Args:
            data_dir: Directory for storing processed VCF data
            temp_dir: Directory for temporary files during processing
        """
        self.data_dir = Path(data_dir)
        self.temp_dir = Path(temp_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session for API calls
        self.session = self._create_request_session()
        
    def _create_request_session(self):
        """Create a requests session with retry logic for API calls."""
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
        
    def anonymize_patient_id(self, patient_id: str) -> str:
        """
        Anonymize patient ID using a one-way hash function.
        
        Args:
            patient_id: Original patient identifier
            
        Returns:
            Anonymized patient identifier
        """
        if not patient_id:
            return f"anon_{uuid.uuid4().hex[:8]}"
            
        # Create a hash of the patient ID
        hash_obj = hashlib.sha256(patient_id.encode())
        return f"anon_{hash_obj.hexdigest()[:8]}"
        
    def parse_vcf(self, vcf_file_path: str) -> List[Dict[str, Any]]:
        """
        Parse VCF file and extract variant information using pysam.
        
        Args:
            vcf_file_path: Path to the VCF file
            
        Returns:
            List of variant dictionaries
        """
        variants = []
        
        try:
            # Use pysam to parse the VCF file
            if 'pysam' in globals():
                vcf_reader = pysam.VariantFile(vcf_file_path)
                
                for record in vcf_reader:
                    # Extract the first ALT allele if available
                    alt = record.alts[0] if record.alts else None
                    
                    variant = {
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'id': record.id,
                        'ref': record.ref,
                        'alt': alt,
                        'qual': record.qual,
                        'filter': ','.join(record.filter.keys()) if record.filter else 'PASS'
                    }
                    variants.append(variant)
            else:
                # Fallback to simple parsing if pysam is not available
                with open(vcf_file_path, 'r') as vcf_file:
                    for line in vcf_file:
                        if line.startswith('#'):
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) >= 5:
                            variant = {
                                'chrom': parts[0],
                                'pos': int(parts[1]),
                                'id': parts[2] if parts[2] != '.' else None,
                                'ref': parts[3],
                                'alt': parts[4],
                                'qual': float(parts[5]) if parts[5] != '.' and len(parts) > 5 else None,
                                'filter': parts[6] if len(parts) > 6 else 'PASS'
                            }
                            variants.append(variant)
        except Exception as e:
            logger.error(f"Error parsing VCF file: {e}")
            raise ValueError(f"Failed to parse VCF file: {e}")
            
        return variants
        
    def determine_variant_types(self, variants: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Determine the types of variants (SNP, indel, SV).
        
        Args:
            variants: List of variant dictionaries
            
        Returns:
            Dictionary with counts of each variant type
        """
        variant_types = {'SNP': 0, 'INDEL': 0, 'SV': 0}
        
        for variant in variants:
            ref = variant['ref']
            alt = variant['alt']
            
            # Single nucleotide polymorphism
            if len(ref) == 1 and len(alt) == 1:
                variant_types['SNP'] += 1
            # Structural variant (arbitrary threshold)
            elif len(ref) > 50 or len(alt) > 50:
                variant_types['SV'] += 1
            # Insertion or deletion
            else:
                variant_types['INDEL'] += 1
                
        return variant_types
              
    def query_dbsnp(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Match variants with dbSNP database and fetch additional information.
        
        Args:
            variants: List of variant dictionaries
            
        Returns:
            Dictionary with counts of matched and unmatched variants and additional info
        """
        matched = 0
        unmatched = 0
        matched_variants = []
        
        for variant in variants:
            if variant['id'] and variant['id'].startswith('rs'):
                matched += 1
                # For variants with rs IDs, we could fetch additional info from NCBI
                # Here we'll just record the ID for now
                matched_variants.append({
                    'id': variant['id'],
                    'chrom': variant['chrom'],
                    'pos': variant['pos'],
                    'ref': variant['ref'],
                    'alt': variant['alt']
                })
            else:
                unmatched += 1
        
        # For real implementation, we could batch query the NCBI E-utilities API
        # Example: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&id=123456&retmode=json
                
        return {
            'matched': matched,
            'unmatched': unmatched,
            'matched_variants': matched_variants[:10]  # Return first 10 for brevity
        }
        
    def calculate_prs(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate polygenic risk score based on variants.
        
        Args:
            variants: List of variant dictionaries
            
        Returns:
            Dictionary with PRS score and risk assessment
        """
        # Define a simple scoring system based on chromosomal locations and variant types
        # In a real implementation, this would use published weights from GWAS studies
        
        # Define some key regions associated with common conditions
        key_regions = {
            # BRCA1 region (chr17)
            '17': [(41196312, 41277500, 'BRCA1', 0.8)],  # (start, end, gene, weight)
            # BRCA2 region (chr13)
            '13': [(32889611, 32973805, 'BRCA2', 0.8)],
            # LDLR region (chr19) - associated with familial hypercholesterolemia
            '19': [(11200000, 11244000, 'LDLR', 0.7)],
            # APOB region (chr2) - associated with familial hypercholesterolemia
            '2': [(21224000, 21266000, 'APOB', 0.6)],
            # PCSK9 region (chr1) - associated with familial hypercholesterolemia
            '1': [(55505000, 55530000, 'PCSK9', 0.5)]
        }
        
        # Calculate weighted score
        total_score = 0
        max_possible_score = 0
        contributing_variants = []
        
        for variant in variants:
            chrom = variant['chrom']
            pos = variant['pos']
            
            # Check if variant is in a key region
            if chrom in key_regions:
                for start, end, gene, weight in key_regions[chrom]:
                    if start <= pos <= end:
                        # Add to score based on variant type
                        # SNPs get full weight, indels get 1.5x weight
                        is_indel = len(variant['ref']) != len(variant['alt'])
                        variant_weight = weight * (1.5 if is_indel else 1.0)
                        
                        total_score += variant_weight
                        max_possible_score += weight
                        
                        contributing_variants.append({
                            'id': variant['id'],
                            'gene': gene,
                            'weight': variant_weight,
                            'is_indel': is_indel
                        })
        
        # Normalize score to 0-1 range if we have any key variants
        if max_possible_score > 0:
            normalized_score = total_score / max_possible_score
        else:
            normalized_score = 0
            
        # Determine risk level
        risk_level = "low"
        if normalized_score > 0.7:
            risk_level = "high"
        elif normalized_score > 0.3:
            risk_level = "moderate"
            
        # Determine most likely condition based on contributing variants
        condition = "unknown"
        if any(v['gene'] in ['BRCA1', 'BRCA2'] for v in contributing_variants):
            condition = "hereditary breast and ovarian cancer"
        elif any(v['gene'] in ['LDLR', 'APOB', 'PCSK9'] for v in contributing_variants):
            condition = "familial hypercholesterolemia"
            
        return {
            'score': round(normalized_score, 2),
            'risk_level': risk_level,
            'condition': condition,
            'contributing_variants': contributing_variants[:5]  # Return top 5 for brevity
        }
        
    def analyze_pathways(self, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform pathway analysis to identify affected biological pathways.
        
        Args:
            variants: List of variant dictionaries
            
        Returns:
            Dictionary with affected pathways and genes
        """
        # Define gene regions and their associated pathways
        gene_regions = {
            # Gene: [chrom, start, end, pathways]
            'LDLR': ['19', 11200000, 11244000, [
                {'name': 'Cholesterol metabolism', 'database': 'KEGG', 'id': 'hsa04979'},
                {'name': 'LDL clearance', 'database': 'Reactome', 'id': 'R-HSA-8964038'}
            ]],
            'APOB': ['2', 21224000, 21266000, [
                {'name': 'Cholesterol metabolism', 'database': 'KEGG', 'id': 'hsa04979'},
                {'name': 'Lipid transport', 'database': 'Reactome', 'id': 'R-HSA-8964058'}
            ]],
            'PCSK9': ['1', 55505000, 55530000, [
                {'name': 'Cholesterol metabolism', 'database': 'KEGG', 'id': 'hsa04979'}
            ]],
            'BRCA1': ['17', 41196312, 41277500, [
                {'name': 'DNA repair', 'database': 'Reactome', 'id': 'R-HSA-73894'},
                {'name': 'Homologous recombination', 'database': 'KEGG', 'id': 'hsa03440'}
            ]],
            'BRCA2': ['13', 32889611, 32973805, [
                {'name': 'DNA repair', 'database': 'Reactome', 'id': 'R-HSA-73894'},
                {'name': 'Homologous recombination', 'database': 'KEGG', 'id': 'hsa03440'}
            ]],
            'TP53': ['17', 7668402, 7687550, [
                {'name': 'p53 signaling pathway', 'database': 'KEGG', 'id': 'hsa04115'},
                {'name': 'Apoptosis', 'database': 'Reactome', 'id': 'R-HSA-109581'}
            ]]
        }
        
        # Track affected genes and pathways
        affected_genes = set()
        pathway_dict = {}  # To avoid duplicates
        
        # Check each variant against gene regions
        for variant in variants:
            chrom = variant['chrom']
            pos = variant['pos']
            
            for gene, (gene_chrom, start, end, pathways) in gene_regions.items():
                if chrom == gene_chrom and start <= pos <= end:
                    affected_genes.add(gene)
                    
                    # Add all pathways for this gene
                    for pathway in pathways:
                        pathway_id = pathway['id']
                        if pathway_id not in pathway_dict:
                            pathway_dict[pathway_id] = {
                                'name': pathway['name'],
                                'database': pathway['database'],
                                'id': pathway_id,
                                'affected_genes': []
                            }
                        
                        if gene not in pathway_dict[pathway_id]['affected_genes']:
                            pathway_dict[pathway_id]['affected_genes'].append(gene)
        
        # Convert to list
        pathways = list(pathway_dict.values())
        
        # If no pathways found, return empty list
        if not pathways:
            return {'affected_pathways': []}
            
        return {
            'affected_pathways': pathways,
            'affected_genes_count': len(affected_genes),
            'affected_genes': list(affected_genes)
        }