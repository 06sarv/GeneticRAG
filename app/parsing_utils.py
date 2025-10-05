from typing import Dict, Any, List, Optional

# --- VEP Consequence Severity Ranking --- #
# This order is based on a general understanding of variant impact, from most to least severe.
# This can be customized based on specific clinical interpretation guidelines.
VEP_CONSEQUENCE_SEVERITY = {
    "transcript_ablation": 1,
    "splice_acceptor_variant": 2,
    "splice_donor_variant": 3,
    "stop_gained": 4,
    "frameshift_variant": 5,
    "stop_lost": 6,
    "start_lost": 7,
    "transcript_amplification": 8,
    "inframe_insertion": 9,
    "inframe_deletion": 10,
    "missense_variant": 11,
    "protein_altering_variant": 12,
    "splice_region_variant": 13,
    "incomplete_terminal_codon_variant": 14,
    "start_retained_variant": 15,
    "stop_retained_variant": 16,
    "synonymous_variant": 17,
    "coding_sequence_variant": 18,
    "mature_miRNA_variant": 19,
    "5_prime_UTR_variant": 20,
    "3_prime_UTR_variant": 21,
    "non_coding_transcript_exon_variant": 22,
    "intron_variant": 23,
    "NMD_transcript_variant": 24,
    "non_coding_transcript_variant": 25,
    "upstream_gene_variant": 26,
    "downstream_gene_variant": 27,
    "TFBS_ablation": 28,
    "TFBS_amplification": 29,
    "regulatory_region_ablation": 30,
    "regulatory_region_amplification": 31,
    "feature_elongation": 32,
    "feature_truncation": 33,
    "intergenic_variant": 34,
    "regulatory_region_variant": 35,
    "TF_binding_site_variant": 36,
    "miRNA": 37,
    "lincRNA": 38,
    "antisense": 39,
    "bidirectional_promoter_lncRNA": 40,
    "retained_intron": 41,
    "sense_intronic": 42,
    "sense_overlapping": 43,
    "3prime_overlapping_ncrna": 44,
    "5prime_overlapping_ncrna": 45,
    "nc_transcript_variant": 46,
    "gene_variant": 47,
    "exon_variant": 48,
    "conserved_intergenic_variant": 49,
    "conserved_intron_variant": 50,
    "rare_amino_acid_variant": 51,
    "initiator_codon_variant": 52,
    "stop_codon_read_through_variant": 53,
    "protein_coding_variant": 54,
    "non_coding_variant": 55,
    "RNA_variant": 56,
    "pseudogene": 57,
    "processed_transcript": 58,
    "rRNA": 59,
    "snoRNA": 60,
    "snRNA": 61,
    "unknown": 999 # Assign a very low priority for unknown consequences
}

def rank_vep_consequences(consequences: List[str]) -> List[str]:
    return sorted(consequences, key=lambda x: VEP_CONSEQUENCE_SEVERITY.get(x, 999))

# --- VEP Record Parsing --- #
def parse_vep_data(vep_response: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed_data = {
        "variant_id": None,
        "most_severe_consequence": None,
        "gene_impacts": []
    }

    if not vep_response:
        return parsed_data

    # Assuming the first entry is the primary variant of interest
    variant = vep_response[0]
    parsed_data["variant_id"] = variant.get("input")
    parsed_data["most_severe_consequence"] = variant.get("most_severe_consequence")

    if "transcript_consequences" in variant:
        for tc in variant["transcript_consequences"]:
            gene_impact = {
                "gene_symbol": tc.get("gene_symbol"),
                "consequence_terms": tc.get("consequence_terms"),
                "impact": tc.get("impact"),
                "hgvsc": tc.get("hgvsc"),
                "hgvsp": tc.get("hgvsp"),
                "sift": tc.get("sift_prediction"),
                "polyphen": tc.get("polyphen_prediction"),
            }
            parsed_data["gene_impacts"].append(gene_impact)

    return parsed_data

# --- MyVariant.info Record Parsing --- #
def parse_myvariant_record(record: Dict[str, Any]) -> Dict[str, Any]:
    parsed_data = {
        "_id": record.get("_id"),
        "dbsnp": record.get("dbsnp", {}),
        "clinvar": record.get("clinvar", {}),
        "cadd": record.get("cadd", {}),
        "gnomad_exome": record.get("gnomad_exome", {}),
        "gnomad_genome": record.get("gnomad_genome", {}),
        "exac": record.get("exac", {}),
        "allele_frequencies": {},
        "annotations": []
    }

    # Extract allele frequencies from various sources
    if "gnomad_exome" in record and "af" in record["gnomad_exome"]:
        parsed_data["allele_frequencies"]["gnomad_exome_af"] = record["gnomad_exome"]["af"]
    if "gnomad_genome" in record and "af" in record["gnomad_genome"]:
        parsed_data["allele_frequencies"]["gnomad_genome_af"] = record["gnomad_genome"]["af"]
    if "exac" in record and "af" in record["exac"]:
        parsed_data["allele_frequencies"]["exac_af"] = record["exac"]["af"]

    # Extract annotations (e.g., from VEP or dbSNP)
    if "snpeff" in record and "ann" in record["snpeff"]:
        for ann in record["snpeff"]["ann"]:
            parsed_data["annotations"].append({
                "gene": ann.get("gene"),
                "effect": ann.get("effect"),
                "impact": ann.get("impact"),
                "hgvs_c": ann.get("hgvs_c"),
                "hgvs_p": ann.get("hgvs_p"),
            })
    elif "vep" in record and "transcript_consequences" in record["vep"]:
        for tc in record["vep"]["transcript_consequences"]:
            parsed_data["annotations"].append({
                "gene": tc.get("gene_symbol"),
                "effect": ", ".join(tc.get("consequence_terms", [])),
                "impact": tc.get("impact"),
                "hgvs_c": tc.get("hgvsc"),
                "hgvs_p": tc.get("hgvsp"),
            })

    return parsed_data

# --- ClinGen Allele Registry Record Parsing --- #
def parse_clingen_allele_registry_record(record: Dict[str, Any]) -> Dict[str, Any]:
    parsed_data = {
        "allele_id": record.get("allele_id"),
        "hgvs": record.get("hgvs"),
        "genomic_coordinates": record.get("genomic_coordinates"),
        "gene_symbol": record.get("gene_symbol"),
        "clinical_significance": record.get("clinical_significance"),
        "conditions": record.get("conditions"),
        "review_status": record.get("review_status"),
        "last_evaluated": record.get("last_evaluated"),
    }
    return parsed_data