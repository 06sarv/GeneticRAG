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
    
    INDIA_ALLELE_FINDER_DIR = Path(__file__).parent / "india_allele_finder_tool"
    INDIA_ALLELE_ANNOTATOR_SCRIPT = INDIA_ALLELE_FINDER_DIR / "indiaAlleleAnnotator.py"
    IAF_FREQ_FILE = INDIA_ALLELE_FINDER_DIR / "iafFreq.txt"

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
        
    def parse_vcf(self, vcf_file_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse VCF file and extract variant information using pysam.

        Args:
            vcf_file_path: Path to the VCF file
            output_dir: Optional directory to save the annotated VCF file. If None, annotation will not be performed.
            
        Returns:
            List of variant dictionaries
        """
        variants = []

        # If output_dir is provided, annotate the VCF file first
        if output_dir:
            annotated_vcf_path = Path(output_dir) / f"annotated_{Path(vcf_file_path).name}"
            try:
                vcf_file_path = self.annotate_with_india_allele_finder(vcf_file_path, str(annotated_vcf_path))
                logger.info(f"VCF file successfully annotated and saved to {vcf_file_path}")
            except Exception as e:
                logger.error(f"Failed to annotate VCF file with India Allele Finder: {e}")
                # Continue with original VCF if annotation fails, or raise error
                # For now, we'll re-raise to ensure annotation is critical
                raise

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

    def query_gnomad(self, variant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Simulate querying gnomAD for variant frequency and information.
        In a real-world scenario, this would involve API calls to gnomAD or querying a local gnomAD database.

        Args:
            variant: A dictionary containing variant information (e.g., chrom, pos, ref, alt).

        Returns:
            A dictionary with gnomAD information if found, otherwise None.
        """
        # This is a placeholder. In a real implementation, you would make an API call.
        # Example gnomAD API endpoint (hypothetical):
        # gnomad_api_url = f"https://gnomad.broadinstitute.org/api?variant={variant['chrom']}-{variant['pos']}-{variant['ref']}-{variant['alt']}"
        # response = self.session.get(gnomad_api_url)
        # if response.status_code == 200:
        #     return response.json()
        # else:
        #     return None

        # For demonstration, we'll simulate some common/rare variants
        # A real implementation would parse the gnomAD response for allele frequencies
        variant_key = f"{variant['chrom']}-{variant['pos']}-{variant['ref']}-{variant['alt']}"

        # Simulate some common variants
        if variant_key == "1-100000-A-G":
            return {"allele_frequency": 0.35, "filter": "PASS", "rs_id": "rs12345"}
        elif variant_key == "X-150000-C-T":
            return {"allele_frequency": 0.12, "filter": "PASS", "rs_id": "rs67890"}
        # Simulate a rare variant
        elif variant_key == "17-41200000-G-A": # Near BRCA1
            return {"allele_frequency": 0.00005, "filter": "PASS", "rs_id": "rs98765"}
        # Simulate a novel variant (not found in gnomAD)
        else:
            return None

    def get_variants_from_vcf(self, vcf_path: str) -> List[Dict[str, Any]]:
        """
        Parses a VCF file and extracts variant information.

        Args:
            vcf_path (str): The path to the VCF file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                   represents a variant with its details.
        """
        variants = []
        try:
            with pysam.VariantFile(vcf_path, "r") as vcf_file:
                for record in vcf_file:
                    # Extract basic variant information
                    variant_info = {
                        "CHROM": record.chrom,
                        "POS": record.pos,
                        "ID": record.id if record.id else ".",
                        "REF": record.ref,
                        "ALT": [str(alt) for alt in record.alts if alt is not None],
                        "QUAL": record.qual if record.qual is not None else ".",
                        "FILTER": str(list(record.filter)) if record.filter else "PASS",
                        "INFO": {key: record.info[key] for key in record.info}
                    }
                    variants.append(variant_info)
        except Exception as e:
            self.logger.error(f"Error reading VCF file {vcf_path}: {e}")
            raise
        return variants

    def interpret_variant_acmg(self, variant: Dict, functional_predictions: Dict, gnomad_allele_frequency: float) -> Dict:
        """
        Interprets a variant based on a simplified ACMG guideline approach.
        This is a simplified model and does not encompass the full complexity of ACMG guidelines.

        Args:
            variant (Dict): The variant information.
            functional_predictions (Dict): Dictionary of functional prediction scores (SIFT, PolyPhen, CADD, REVEL).
            gnomad_allele_frequency (float): Allele frequency from gnomAD.

        Returns:
            Dict: A dictionary containing the ACMG classification and supporting evidence.
        """
        classification = "Uncertain significance"
        evidence = []

        # Apply simplified ACMG-like rules

        # Population Data (PM2 - Absent in control population / BA1 - Benign allele frequency)
        if gnomad_allele_frequency < 0.001:  # Very rare in gnomAD
            evidence.append("PM2: Absent/very rare in gnomAD population (AF < 0.001)")
            if gnomad_allele_frequency == 0: # Not observed
                classification = "Likely pathogenic"
        elif gnomad_allele_frequency > 0.05: # Common in gnomAD
            evidence.append("BA1: Common in gnomAD population (AF > 0.05)")
            classification = "Benign"

        # Functional Data (PS3 - Deleterious in functional studies / BP4 - Benign in functional studies)
        sift_score = functional_predictions.get("sift_score", 0.5)
        polyphen_score = functional_predictions.get("polyphen_score", 0.5)
        cadd_score = functional_predictions.get("cadd_score", 10.0)
        revel_score = functional_predictions.get("revel_score", 0.5)

        if sift_score < 0.05 and polyphen_score > 0.9 and cadd_score > 20 and revel_score > 0.7: # Multiple damaging predictions
            evidence.append("PS3: Multiple in-silico tools predict deleterious effect")
            if classification == "Uncertain significance":
                classification = "Likely pathogenic"
            elif classification == "Likely benign":
                classification = "Uncertain significance"
        elif sift_score > 0.8 and polyphen_score < 0.1 and cadd_score < 5 and revel_score < 0.3: # Multiple benign predictions
            evidence.append("BP4: Multiple in-silico tools predict benign effect")
            if classification == "Uncertain significance":
                classification = "Likely benign"
            elif classification == "Likely pathogenic":
                classification = "Uncertain significance"

        # Example for a strong pathogenic criterion (PVS1 - Null variant in a gene where LOF is a known mechanism of disease)
        # This would require gene-specific knowledge, which is beyond the scope of this simulation.
        # For demonstration, let's assume a specific variant is known to be PVS1
        variant_key = f"{variant.get('CHROM')}-{variant.get('POS')}-{variant.get('REF')}-{variant.get('ALT')}"
        if "chr1-1000-A-G" in variant_key: # Example of a variant that might be PVS1
            evidence.append("PVS1: Null variant (simulated) in a gene where LOF is a known mechanism of disease")
            classification = "Pathogenic"

        return {
            "acmg_classification": classification,
            "supporting_evidence": evidence
        }

    def perform_network_pharmacology_analysis(self, variant: dict) -> dict:
        """
        Simulates network pharmacology analysis for a given variant.
        This method identifies genes associated with the variant and then
        simulates interactions with drugs or pathways.

        Args:
            variant (dict): A dictionary containing variant information.

        Returns:
            dict: A dictionary containing simulated network pharmacology analysis results.
        """
        variant_key = f"{variant['CHROM']}:{variant['POS']}:{variant['REF']}:{variant['ALT']}"
        
        # Simulate gene association based on variant key
        simulated_gene_associations = {
            "chr1:100:A:G": ["GENE_A", "GENE_B"],
            "chr2:200:C:T": ["GENE_C"],
            "chr3:300:G:A": ["GENE_D", "GENE_E", "GENE_F"],
        }
        
        associated_genes = simulated_gene_associations.get(variant_key, [f"UNKNOWN_GENE_{variant_key.split(':')[0]}"])

        # Simulate drug and pathway interactions for associated genes
        simulated_interactions = {}
        for gene in associated_genes:
            simulated_interactions[gene] = {
                "drugs": [f"DRUG_{gene}_1", f"DRUG_{gene}_2"],
                "pathways": [f"PATHWAY_{gene}_X", f"PATHWAY_{gene}_Y"]
            }
        
        return {
            "variant_key": variant_key,
            "associated_genes": associated_genes,
            "simulated_interactions": simulated_interactions
        }

    def predict_functional_impact(self, variant: Dict) -> Dict:
        """
        Simulates querying functional prediction tools (SIFT, PolyPhen, CADD, REVEL)
        for a given variant and returns their scores.

        Args:
            variant (Dict): A dictionary representing the variant, typically
                            containing 'CHROM', 'POS', 'REF', 'ALT' keys.

        Returns:
            Dict: A dictionary containing simulated functional prediction scores.
        """
        # In a real-world scenario, this would involve API calls to external databases
        # or running local prediction tools. For this simulation, we return dummy scores.
        variant_key = f"{variant.get('CHROM')}-{variant.get('POS')}-{variant.get('REF')}-{variant.get('ALT')}"

        # Simulate scores based on some arbitrary logic or pre-defined values
        # For demonstration, let's make some variants appear "damaging" and others "benign"
        if "chr1-1000-A-G" in variant_key: # Example of a "damaging" variant
            sift_score = 0.01 # Lower score indicates damaging
            polyphen_score = 0.95 # Higher score indicates probably damaging
            cadd_score = 25.0 # Higher score indicates more deleterious
            revel_score = 0.90 # Higher score indicates more deleterious
        elif "chr2-2000-C-T" in variant_key: # Example of a "benign" variant
            sift_score = 0.85
            polyphen_score = 0.05
            cadd_score = 1.5
            revel_score = 0.10
        else: # Default scores for other variants
            sift_score = 0.5
            polyphen_score = 0.5
            cadd_score = 10.0
            revel_score = 0.5

        return {
            "sift_score": sift_score,
            "polyphen_score": polyphen_score,
            "cadd_score": cadd_score,
            "revel_score": revel_score,
        }

    def identify_novel_recurrent_variants(self, vcf_path: str) -> Dict:
        """
        Identifies novel and recurrent variants in a VCF file by comparing them
        against a simulated gnomAD population database.
        """
        novel_variants = []
        recurrent_variants = []

        for variant in variants:
            gnomad_info = self.query_gnomad(variant)

            if gnomad_info is None:
                # Variant not found in gnomAD, consider it novel
                novel_variants.append(variant)
            else:
                # Variant found in gnomAD, check allele frequency
                # Define a threshold for 'recurrent' (e.g., AF > 0.01)
                if gnomad_info.get("allele_frequency", 0) > 0.01:
                    recurrent_variants.append({
                        **variant,
                        "gnomad_af": gnomad_info["allele_frequency"],
                        "gnomad_rs_id": gnomad_info.get("rs_id")
                    })
                else:
                    # If found but rare, still consider it for further analysis, but not 'recurrent' by this definition
                    pass # Or add to another category like 'known_rare_variants'

        return {
            "novel_variants": novel_variants,
            "recurrent_variants": recurrent_variants
        }

    def annotate_with_india_allele_finder(self, vcf_file_path: str, output_vcf_path: str) -> str:
        """
        Annotate VCF file with India Allele Finder frequencies.

        Args:
            vcf_file_path: Path to the input VCF file.
            output_vcf_path: Path to save the annotated VCF file.

        Returns:
            Path to the annotated VCF file.
        """
        if not self.INDIA_ALLELE_ANNOTATOR_SCRIPT.exists():
            raise FileNotFoundError(f"India Allele Annotator script not found at {self.INDIA_ALLELE_ANNOTATOR_SCRIPT}")
        if not self.IAF_FREQ_FILE.exists():
            raise FileNotFoundError(f"India Allele Frequencies file not found at {self.IAF_FREQ_FILE}")

        command = [
            "python3",
            str(self.INDIA_ALLELE_ANNOTATOR_SCRIPT),
            "-i", str(vcf_file_path),
            "-o", str(output_vcf_path),
            "-f", str(self.IAF_FREQ_FILE)
        ]

        try:
            logger.info(f"Running India Allele Finder annotation: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"India Allele Finder stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"India Allele Finder stderr: {result.stderr}")
            return output_vcf_path
        except subprocess.CalledProcessError as e:
            logger.error(f"India Allele Finder annotation failed: {e.stderr}")
            raise RuntimeError(f"India Allele Finder annotation failed: {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during India Allele Finder annotation: {e}")
            raise RuntimeError(f"An unexpected error occurred during India Allele Finder annotation: {e}")