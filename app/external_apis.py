import requests
import json
import time
from typing import Dict, Any, List, Optional

# --- ClinGen Allele Registry API --- #
class ClinGenAlleleRegistryAPI:
    def __init__(self, base_url="https://reg.clingen.me/api/external_allele_registry"): # type: ignore
        self.base_url = base_url

    def get_allele_data(self, hgvs_expression: str) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.base_url}/alleles"
        params = {"hgvs": hgvs_expression}
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying ClinGen Allele Registry: {e}")
            return None

# --- MyVariant.info API --- #
class MyVariantInfoAPI:
    def __init__(self, base_url="https://myvariant.info/v1/variant"): # type: ignore
        self.base_url = base_url

    def get_variant_data(self, rsid: str) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(f"{self.base_url}/{rsid}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying MyVariant.info: {e}")
            return None

# --- VEP API --- #
VEP_API_URL = "https://rest.ensembl.org/vep/human/hgvs/"

def query_vep(hgvs_expression: str) -> Optional[List[Dict[str, Any]]]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = {"hgvs_notations": [hgvs_expression]}
    try:
        response = requests.post(VEP_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying VEP: {e}")
        return None

def get_vep_consequences(vep_data: Dict[str, Any]) -> List[str]:
    consequences = set()
    if "transcript_consequences" in vep_data:
        for tc in vep_data["transcript_consequences"]:
            if "consequence_terms" in tc:
                consequences.update(tc["consequence_terms"])
    if "intergenic_consequences" in vep_data:
        for ic in vep_data["intergenic_consequences"]:
            if "consequence_terms" in ic:
                consequences.update(ic["consequence_terms"])
    if "regulatory_feature_consequences" in vep_data:
        for rfc in vep_data["regulatory_feature_consequences"]:
            if "consequence_terms" in rfc:
                consequences.update(rfc["consequence_terms"])
    if "motif_feature_consequences" in vep_data:
        for mfc in vep_data["motif_feature_consequences"]:
            if "consequence_terms" in mfc:
                consequences.update(mfc["consequence_terms"])
    return sorted(list(consequences))

def get_vep_most_severe_consequence(vep_data: Dict[str, Any]) -> Optional[str]:
    consequences = get_vep_consequences(vep_data)
    if not consequences:
        return None
    # This is a simplified approach. A more robust solution would use a predefined order of severity.
    return consequences[0] # Just taking the first one for now

def get_vep_transcript_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("transcript_consequences", [])

def get_vep_regulatory_feature_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("regulatory_feature_consequences", [])

def get_vep_intergenic_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("intergenic_consequences", [])

def get_vep_motif_feature_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("motif_feature_consequences", [])

def get_vep_gene_fusion_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("gene_fusion_consequences", [])

def get_vep_structural_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("structural_variant_consequences", [])

def get_vep_copy_number_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("copy_number_variant_consequences", [])

def get_vep_protein_change_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("protein_change_consequences", [])

def get_vep_regulatory_region_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("regulatory_region_consequences", [])

def get_vep_mirna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("mirna_consequences", [])

def get_vep_lincrna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("lincrna_consequences", [])

def get_vep_rrna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("rrna_consequences", [])

def get_vep_snrna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("snrna_consequences", [])

def get_vep_snorna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("snorna_consequences", [])

def get_vep_pseudogene_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("pseudogene_consequences", [])

def get_vep_processed_transcript_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("processed_transcript_consequences", [])

def get_vep_retained_intron_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("retained_intron_consequences", [])

def get_vep_bidirectional_promoter_lncrna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("bidirectional_promoter_lncrna_consequences", [])

def get_vep_3prime_overlapping_ncrna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("3prime_overlapping_ncrna_consequences", [])

def get_vep_5prime_overlapping_ncrna_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("5prime_overlapping_ncrna_consequences", [])

def get_vep_sense_intronic_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("sense_intronic_consequences", [])

def get_vep_sense_overlapping_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("sense_overlapping_consequences", [])

def get_vep_antisense_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("antisense_consequences", [])

def get_vep_non_coding_transcript_exon_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("non_coding_transcript_exon_variant_consequences", [])

def get_vep_non_coding_transcript_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("non_coding_transcript_variant_consequences", [])

def get_vep_mature_mirna_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("mature_mirna_variant_consequences", [])

def get_vep_start_lost_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("start_lost_consequences", [])

def get_vep_stop_gained_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("stop_gained_consequences", [])

def get_vep_stop_lost_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("stop_lost_consequences", [])

def get_vep_frameshift_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("frameshift_variant_consequences", [])

def get_vep_inframe_insertion_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("inframe_insertion_consequences", [])

def get_vep_inframe_deletion_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("inframe_deletion_consequences", [])

def get_vep_missense_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("missense_variant_consequences", [])

def get_vep_splice_donor_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("splice_donor_variant_consequences", [])

def get_vep_splice_acceptor_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("splice_acceptor_variant_consequences", [])

def get_vep_splice_region_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("splice_region_variant_consequences", [])

def get_vep_synonymous_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("synonymous_variant_consequences", [])

def get_vep_coding_sequence_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("coding_sequence_variant_consequences", [])

def get_vep_5_prime_utr_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("5_prime_utr_variant_consequences", [])

def get_vep_3_prime_utr_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("3_prime_utr_variant_consequences", [])

def get_vep_intron_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("intron_variant_consequences", [])

def get_vep_upstream_gene_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("upstream_gene_variant_consequences", [])

def get_vep_downstream_gene_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("downstream_gene_variant_consequences", [])

def get_vep_intergenic_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("intergenic_variant_consequences", [])

def get_vep_regulatory_region_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("regulatory_region_variant_consequences", [])

def get_vep_tfbs_ablation_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("tfbs_ablation_consequences", [])

def get_vep_tfbs_amplification_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("tfbs_amplification_consequences", [])

def get_vep_feature_elongation_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("feature_elongation_consequences", [])

def get_vep_feature_truncation_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("feature_truncation_consequences", [])

def get_vep_regulatory_feature_ablation_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("regulatory_feature_ablation_consequences", [])

def get_vep_regulatory_feature_amplification_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("regulatory_feature_amplification_consequences", [])

def get_vep_motif_feature_ablation_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("motif_feature_ablation_consequences", [])

def get_vep_motif_feature_amplification_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("motif_feature_amplification_consequences", [])

def get_vep_transcript_ablation_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("transcript_ablation_consequences", [])

def get_vep_transcript_amplification_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("transcript_amplification_consequences", [])

def get_vep_gene_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("gene_variant_consequences", [])

def get_vep_start_retained_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("start_retained_variant_consequences", [])

def get_vep_stop_retained_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("stop_retained_variant_consequences", [])

def get_vep_initiator_codon_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("initiator_codon_variant_consequences", [])

def get_vep_stop_codon_read_through_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("stop_codon_read_through_variant_consequences", [])

def get_vep_nc_transcript_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("nc_transcript_variant_consequences", [])

def get_vep_exon_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("exon_variant_consequences", [])

def get_vep_conserved_intergenic_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("conserved_intergenic_variant_consequences", [])

def get_vep_conserved_intron_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("conserved_intron_variant_consequences", [])

def get_vep_rare_amino_acid_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("rare_amino_acid_variant_consequences", [])

def get_vep_protein_altering_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("protein_altering_variant_consequences", [])

def get_vep_incomplete_terminal_codon_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("incomplete_terminal_codon_variant_consequences", [])

def get_vep_coding_transcript_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("coding_transcript_variant_consequences", [])

def get_vep_non_coding_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("non_coding_variant_consequences", [])

def get_vep_rna_variant_consequences(vep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return vep_data.get("rna_variant_consequences", [])