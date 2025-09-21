#!/usr/bin/env python3
"""
Test script for VCF processing functionality.
"""
import requests
import json
import os
import sys

# Configuration
API_URL = "http://localhost:8000/api/process_vcf"
VCF_FILE_PATH = os.path.abspath("data/test/sample.vcf")
PATIENT_ID = "TEST123"

def test_vcf_processing():
    """Test the VCF processing API endpoint."""
    print(f"Testing VCF processing with file: {VCF_FILE_PATH}")
    
    # Prepare request payload
    payload = {
        "file_path": VCF_FILE_PATH,
        "patient_id": PATIENT_ID,
        "analysis_type": "all"
    }
    
    try:
        # Send request to API
        response = requests.post(API_URL, json=payload)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\n✅ VCF Processing Test Successful!")
            print(f"Anonymized ID: {result['anonymized_id']}")
            
            # Print variant summary
            if "variants" in result["results"]:
                variants = result["results"]["variants"]
                print(f"\nVariant Summary:")
                print(f"  Total variants: {variants['count']}")
                print(f"  Types: {json.dumps(variants['types'], indent=2)}")
            
            # Print dbSNP results
            if "dbsnp" in result["results"]:
                dbsnp = result["results"]["dbsnp"]
                print(f"\ndbSNP Matching:")
                print(f"  Matched: {dbsnp['matched']}")
                print(f"  Unmatched: {dbsnp['unmatched']}")
            
            # Print PRS results
            if "prs" in result["results"]:
                prs = result["results"]["prs"]
                print(f"\nPolygenic Risk Score:")
                print(f"  Score: {prs['score']}")
                print(f"  Risk Level: {prs['risk_level']}")
            
            # Print pathway analysis
            if "pathways" in result["results"]:
                pathways = result["results"]["pathways"]
                print(f"\nPathway Analysis:")
                for pathway in pathways["affected_pathways"]:
                    print(f"  - {pathway['name']} ({pathway['database']})")
                    print(f"    Affected genes: {', '.join(pathway['affected_genes'])}")
            
            return True
        else:
            print(f"\n❌ Test Failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    
    except Exception as e:
        print(f"\n❌ Test Failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if server is running
    try:
        requests.get("http://localhost:8000/docs")
        print("Server is running. Starting test...")
    except requests.exceptions.ConnectionError:
        print("ERROR: Server is not running. Please start the server first with:")
        print("  python run_server.py")
        sys.exit(1)
    
    # Run test
    test_vcf_processing()