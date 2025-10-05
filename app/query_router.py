from enum import Enum
from typing import Optional

class QueryClassification(Enum):
    HGVS = "HGVS"
    RSID = "RSID"
    GENE_DISEASE = "Gene-Disease"
    PHENOTYPE = "Phenotype"
    UNKNOWN = "Unknown"

class GenomicQueryRouter:
    def __init__(self, llm):
        self.llm = llm
        self.query_classifier_prompt = """
        You are an AI assistant that classifies genomic queries into one of the following categories:
        - HGVS: Human Genome Variation Society nomenclature (e.g., "NM_004006.2:c.1582G>A", "NC_000017.10:g.41276104C>T")
        - RSID: Reference SNP cluster ID (e.g., "rs12345")
        - Gene-Disease: Queries linking a gene to a disease (e.g., "BRCA1 and breast cancer", "What diseases are associated with CFTR?")
        - Phenotype: Queries describing a phenotype or symptom (e.g., "short stature", "intellectual disability")
        - Unknown: Any other query that does not fit the above categories.

        Classify the following query:
        Query: {query}
        Classification: """

    def classify_query(self, query: str) -> QueryClassification:
        prompt = self.query_classifier_prompt.format(query=query)
        # Assuming self.llm has a method like .generate_text or similar
        # that takes a prompt and returns a string response.
        # The actual implementation might vary based on the LLM interface.
        response = self.llm.generate_text(prompt)
        response_text = response.strip().upper()

        if "HGVS" in response_text:
            return QueryClassification.HGVS
        elif "RSID" in response_text:
            return QueryClassification.RSID
        elif "GENE-DISEASE" in response_text.replace(" ", "-"):
            return QueryClassification.GENE_DISEASE
        elif "PHENOTYPE" in response_text:
            return QueryClassification.PHENOTYPE
        else:
            return QueryClassification.UNKNOWN

    def route_query(self, query: str):
        classification = self.classify_query(query)
        # In a full implementation, this would route to different handlers
        # based on the classification.
        return {"query": query, "classification": classification.value}