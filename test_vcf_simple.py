#!/usr/bin/env python3
"""
Simple test script for VCF processing functionality.
"""
import os
from app.vcf_processor import VCFProcessor

# Configuration
VCF_FILE_PATH = os.path.abspath("data/test/sample.vcf")
PATIENT_ID = "TEST123"

def test_vcf_processing():
    """Test the VCF processing directly."""
    print(f"Testing VCF processing with file: {VCF_FILE_PATH}")
    
    try:
        # Create VCF processor instance
        vcf_processor = VCFProcessor()
        
        # Test anonymization
        anonymized_id = vcf_processor.anonymize_patient_id(PATIENT_ID)
        print(f"\nAnonymized ID: {anonymized_id}")
        
        # Test VCF parsing
        variants = vcf_processor.parse_vcf(VCF_FILE_PATH)
        print(f"\nParsed {len(variants)} variants from VCF file")
        
        # Test variant type determination
        variant_types = vcf_processor.determine_variant_types(variants)
        print(f"\nVariant Types:")
        for vtype, count in variant_types.items():
            print(f"  {vtype}: {count}")
        
        # Test dbSNP query
        dbsnp_results = vcf_processor.query_dbsnp(variants)
        print(f"\ndbSNP Results:")
        print(f"  Matched: {dbsnp_results['matched']}")
        print(f"  Unmatched: {dbsnp_results['unmatched']}")
        
        # Test PRS calculation
        prs_score = vcf_processor.calculate_prs(variants)
        print(f"\nPRS Score:")
        print(f"  Score: {prs_score['score']}")
        print(f"  Risk Level: {prs_score['risk_level']}")
        
        # Test pathway analysis
        pathway_analysis = vcf_processor.analyze_pathways(variants)
        print(f"\nPathway Analysis:")
        for pathway in pathway_analysis["affected_pathways"]:
            print(f"  - {pathway['name']} ({pathway['database']})")
            print(f"    Affected genes: {', '.join(pathway['affected_genes'])}")
        
        print("\n✅ VCF Processing Test Successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test
    test_vcf_processing()