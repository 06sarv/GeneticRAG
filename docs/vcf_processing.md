# VCF Processing Documentation

## Overview

The VCF (Variant Call Format) processing module enables the genetic chatbot backend to analyze genetic variants from VCF files. This functionality supports:

- Parsing VCF files to identify SNPs, indels, and structural variants
- Matching variants with dbSNP for rs ID identification
- Calculating polygenic risk scores (PRS)
- Performing pathway analysis using KEGG and Reactome databases
- Maintaining patient confidentiality through data anonymization

## API Usage

### Process VCF File

**Endpoint:** `POST /api/process_vcf`

**Request Body:**
```json
{
  "file_path": "/path/to/patient.vcf",
  "patient_id": "PATIENT123",
  "analysis_type": "all"
}
```

**Parameters:**
- `file_path` (required): Path to the VCF file to process
- `patient_id` (optional): Patient identifier (will be anonymized)
- `analysis_type` (optional): Type of analysis to perform
  - Options: "variant", "dbsnp", "prs", "pathway", "all" (default)

**Response:**
```json
{
  "success": true,
  "anonymized_id": "anon_7f8a9b2c",
  "results": {
    "variants": {
      "count": 12500,
      "types": {
        "SNP": 11200,
        "INDEL": 950,
        "SV": 350
      }
    },
    "dbsnp": {
      "matched": 10800,
      "unmatched": 1700
    },
    "prs": {
      "score": 0.68,
      "risk_level": "moderate",
      "condition": "familial hypercholesterolemia"
    },
    "pathways": {
      "affected_pathways": [
        {
          "name": "Cholesterol metabolism",
          "database": "KEGG",
          "id": "hsa04979",
          "affected_genes": ["LDLR", "APOB", "PCSK9"]
        }
      ]
    }
  }
}
```

## Security Considerations

1. **Patient Confidentiality**
   - All patient IDs are anonymized using a one-way hashing algorithm
   - No personally identifiable information is stored or transmitted
   - Results are presented without patient identifiers

2. **Data Handling**
   - VCF files are processed locally without external API calls
   - No raw genetic data is sent to external services
   - Analysis results are stored temporarily and can be purged after use

3. **Access Control**
   - API endpoints require proper authentication
   - Access logs track all VCF processing requests
   - Implement role-based access control for sensitive operations

## Implementation Details

The VCF processing functionality is implemented in two main components:

1. **VCFProcessor Class** (`app/vcf_processor.py`)
   - Core functionality for parsing and analyzing VCF files
   - Methods for variant identification, dbSNP matching, and pathway analysis
   - Secure handling of patient data

2. **API Integration** (`app/main.py`)
   - REST API endpoints for VCF processing
   - Request validation and error handling
   - Response formatting and security measures

## Limitations and Future Work

- Current implementation provides basic PRS calculation; more sophisticated models can be implemented
- Limited to standard VCF format; may need extensions for specialized formats
- Pathway analysis uses simplified approach; can be enhanced with more comprehensive databases
- Consider implementing batch processing for multiple VCF files