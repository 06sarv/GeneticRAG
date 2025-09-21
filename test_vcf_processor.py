#!/usr/bin/env python3
"""
Simple test script for the VCF processor functionality.
"""

import os
import sys
from app.vcf_processor import VCFProcessor

def test_vcf_processor():
    """Test the VCF processor functionality."""
    print("Testing VCF Processor...")
    
    # Path to the sample VCF file
    vcf_file = "data/test/sample.vcf"
    
    # Check if the file exists
    if not os.path.exists(vcf_file):
        print(f"Error: Sample VCF file not found at {vcf_file}")
        return False
    
    # Create VCF processor instance
    processor = VCFProcessor()
    
    # Test anonymization
    patient_id = "PATIENT12345"
    anon_id = processor.anonymize_patient_id(patient_id)
    print(f"\n1. Anonymized patient ID: {anon_id}")
    
    # Test VCF parsing
    print("\n2. Parsing VCF file...")
    variants = processor.parse_vcf(vcf_file)
    print(f"   Found {len(variants)} variants")
    if variants:
        print("   Sample variant:")
        for key, value in variants[0].items():
            print(f"   - {key}: {value}")
    
    # Test variant type determination
    print("\n3. Determining variant types...")
    variant_types = processor.determine_variant_types(variants)
    print("   Variant type counts:")
    for vtype, count in variant_types.items():
        print(f"   - {vtype}: {count}")
    
    # Test dbSNP matching
    print("\n4. Matching with dbSNP...")
    dbsnp_results = processor.query_dbsnp(variants)
    print("   dbSNP matching results:")
    for key, value in dbsnp_results.items():
        print(f"   - {key}: {value}")
    
    # Test PRS calculation
    print("\n5. Calculating polygenic risk score...")
    prs_results = processor.calculate_prs(variants)
    print("   PRS results:")
    for key, value in prs_results.items():
        print(f"   - {key}: {value}")
    
    # Test pathway analysis
    print("\n6. Analyzing biological pathways...")
    pathway_results = processor.analyze_pathways(variants)
    print("   Pathway analysis results:")
    for pathway in pathway_results['affected_pathways']:
        print(f"   - Pathway: {pathway['name']} ({pathway['database']} {pathway['id']})")
        print(f"     Affected genes: {', '.join(pathway['affected_genes'])}")
    
    print("\nVCF processing test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_vcf_processor()
    sys.exit(0 if success else 1)