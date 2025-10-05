from typing import Any, Dict, List, Union
from typing import Any, Dict, List, Union
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from app.llm_utils import load_llm

class PopulationFilterConfig:
    """Filter only redundant population frequencies, keep everything else"""
    
    def __init__(self):
        # CONFIGURE THIS: Which populations matter for your use case
        self.PRIMARY_POPULATIONS = ['sas']  # South Asian (India, Pakistan, Bangladesh, Sri Lanka)
        self.SECONDARY_POPULATIONS = []     # Empty = don't include secondary populations
        self.INCLUDE_GLOBAL = True          # Always show overall allele frequency
        
        # Population codes explained:
        # sas = South Asian, afr = African, eas = East Asian, nfe = Non-Finnish European
        # amr = Admixed American, fin = Finnish, asj = Ashkenazi Jewish, mid = Middle Eastern
        # oth = Other, ami = Amish
        
    def should_include_field(self, field_name: str) -> bool:
        """
        Returns True if field should be included in AI summary.
        Only filters out population-specific frequency columns.
        """
        field_lower = field_name.lower()
        
        # Check if this is a population frequency field
        # Pattern: gnomad_genome_af_af_POPULATION or gnomad_genome_ac_ac_POPULATION etc.
        is_pop_freq = False
        freq_prefixes = [
            'gnomad_genome_af_af_',
            'gnomad_genome_ac_ac_',
            'gnomad_genome_an_an_',
            'gnomad_genome_hom_hom_',
            'gnomad_exome_af_af_',
            'gnomad_exome_ac_ac_',
            'gnomad_exome_an_an_',
            'gnomad_exome_hom_hom_'
        ]
        
        for prefix in freq_prefixes:
            if field_lower.startswith(prefix):
                is_pop_freq = True
                break
        
        # If not a population frequency field, always include
        if not is_pop_freq:
            return True
        
        # For population frequency fields, apply filtering
        
        # Always include global/overall frequencies (fields ending with _af, _ac, _an, _hom)
        if self.INCLUDE_GLOBAL:
            # These are the aggregate frequencies across all populations
            if (field_name.endswith('_af_af') or field_name.endswith('_ac_ac') or 
                field_name.endswith('_an_an') or field_name.endswith('_hom_hom')):
                return True
            # Also catch XX/XY sex-specific global frequencies
            if (field_name.endswith('_af_xx') or field_name.endswith('_af_xy') or
                field_name.endswith('_ac_xx') or field_name.endswith('_ac_xy') or
                field_name.endswith('_an_xx') or field_name.endswith('_an_xy') or
                field_name.endswith('_hom_xx') or field_name.endswith('_hom_xy')):
                return True
        
        # Check if field contains primary populations
        for pop in self.PRIMARY_POPULATIONS:
            if f'_{pop}_' in field_lower or f'_{pop}' in field_lower.split('_')[-1]:
                return True
        
        # Check secondary populations
        for pop in self.SECONDARY_POPULATIONS:
            if f'_{pop}_' in field_lower or f'_{pop}' in field_lower.split('_')[-1]:
                return True
        
        # If we get here, it's a population frequency we don't care about
        return False
    
    def get_category_for_field(self, field_name: str) -> str:
        """Assign semantic category to each field for grouped display"""
        field_lower = field_name.lower()
        
        # Core variant identification
        if field_name in ['rsid', 'variant_id', '_id', 'input', 'query', 'chrom', 'pos', 
                          'ref', 'alt', 'assembly', 'allele_string', 'seq_region', 
                          'start', 'end', 'strand', 'variant_class']:
            return "VARIANT_IDENTIFICATION"
        
        # Gene and transcript information
        if any(x in field_lower for x in ['gene', 'transcript', 'feature', 'biotype', 
                                           'canonical', 'mane', 'tsl', 'appris', 'ccds',
                                           'ensp', 'swissprot', 'trembl', 'uniparc', 'protein_id']):
            return "GENE_AND_TRANSCRIPT"
        
        # HGVS nomenclature
        if 'hgvs' in field_lower:
            return "HGVS_NOMENCLATURE"
        
        # Variant consequences and impact
        if any(x in field_lower for x in ['consequence', 'impact', 'exon', 'intron', 
                                           'amino_acids', 'codons', 'protein_start', 
                                           'protein_end', 'lof']):
            return "VARIANT_CONSEQUENCES"
        
        # Clinical significance (ClinVar)
        if 'clinvar' in field_lower:
            return "CLINICAL_SIGNIFICANCE"
        
        # Functional prediction scores
        if any(x in field_lower for x in ['sift', 'polyphen', 'cadd', 'revel', 'vest', 
                                           'metasvm', 'metalr', 'mcap', 'lrt', 
                                           'mutationtaster', 'mutationassessor', 
                                           'fathmm', 'provean']):
            return "FUNCTIONAL_PREDICTIONS"
        
        # Population frequencies and counts
        if any(x in field_lower for x in ['_af', '_ac', '_an', 'freq', 'hom', 'allele']):
            return "POPULATION_FREQUENCIES"
        
        # Database cross-references
        if any(x in field_lower for x in ['cosmic', 'dbsnp', 'colocated']):
            return "DATABASE_REFERENCES"
        
        # Technical quality metrics
        if any(x in field_lower for x in ['mq', 'ranksum', 'baseq', 'clipping', 'readpos']):
            return "QUALITY_METRICS"
        
        # Everything else
        return "ADDITIONAL_ANNOTATIONS"

POPULATION_FILTER = PopulationFilterConfig()

class EnhancedVariantSummarizer:
    def __init__(self, debug: bool = True):
        """
        Initialize enhanced variant summarizer with comprehensive debugging
        """
        self.debug = debug

        if self.debug:
            print(f"🔧 DEBUG: Initializing EnhancedVariantSummarizer")

        self.model, self.tokenizer = load_llm()

        if self.debug:
            print(f"🔧 DEBUG: Model loaded successfully!")
            try:
                print(f"🔧 DEBUG: Model device: {self.model.device}")
            except Exception:
                pass
            if hasattr(self.model, 'get_memory_footprint'):
                try:
                    print(f"🔧 DEBUG: Model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")
                except Exception:
                    pass
            if torch.cuda.is_available():
                try:
                    print(f"🔧 DEBUG: GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                except Exception:
                    pass

    def format_dataframe_for_prompt(self, df: pd.DataFrame, source_name: str) -> str:
        """Convert DataFrame to filtered, semantically grouped string for AI"""
        if self.debug:
            print(f"🔧 DEBUG: Formatting {source_name} data WITH SMART FILTERING")
            print(f"🔧 DEBUG: DataFrame shape: {df.shape}")

        if df.empty:
            return f"{source_name}: No data available"

        # Get first row
        data_dict = df.iloc[0].to_dict()

        # Filter and categorize fields
        categorized_fields = {}
        total_fields = 0
        kept_fields = 0
        filtered_fields = 0

        for field_name, field_value in data_dict.items():
            total_fields += 1

            # Skip null/empty values
            try:
                if pd.isna(field_value):
                    continue
                if isinstance(field_value, str) and field_value.strip() == '':
                    continue
                if isinstance(field_value, list) and len(field_value) == 0:
                    continue
            except (ValueError, TypeError):
                if field_value is None:
                    continue

            # Apply smart filtering (only filters population frequencies)
            if not POPULATION_FILTER.should_include_field(field_name):
                filtered_fields += 1
                continue

            kept_fields += 1

            # Categorize field
            category = POPULATION_FILTER.get_category_for_field(field_name)

            if category not in categorized_fields:
                categorized_fields[category] = {}

            categorized_fields[category][field_name] = field_value

        if self.debug:
            print(f"🔧 DEBUG: Total fields: {total_fields}")
            print(f"🔧 DEBUG: Non-null fields: {kept_fields + filtered_fields}")
            print(f"🔧 DEBUG: Filtered out: {filtered_fields} (population frequencies)")
            print(f"🔧 DEBUG: Kept for AI: {kept_fields}")
            print(f"🔧 DEBUG: Categories: {list(categorized_fields.keys())}")

        # Format output with semantic grouping
        formatted_sections = []
        formatted_sections.append(f"=== {source_name.upper()} ANNOTATION DATA ===\n")

        # Define logical category order for presentation
        category_order = [
            "VARIANT_IDENTIFICATION",
            "GENE_AND_TRANSCRIPT",
            "HGVS_NOMENCLATURE",
            "VARIANT_CONSEQUENCES",
            "CLINICAL_SIGNIFICANCE",
            "FUNCTIONAL_PREDICTIONS",
            "POPULATION_FREQUENCIES",
            "DATABASE_REFERENCES",
            "QUALITY_METRICS",
            "ADDITIONAL_ANNOTATIONS"
        ]

        for category in category_order:
            if category not in categorized_fields:
                continue

            fields = categorized_fields[category]
            if not fields:
                continue

            # Make category headers readable
            category_display = category.replace('_', ' ')
            formatted_sections.append(f"\n{category_display}:")

            # Sort fields alphabetically within category
            for field_name in sorted(fields.keys()):
                field_value = fields[field_name]

                # Format value based on type
                if isinstance(field_value, list):
                    if len(field_value) <= 3:
                        value_str = ', '.join(str(x) for x in field_value)
                    else:
                        value_str = ', '.join(str(x) for x in field_value[:3])
                        value_str += f" ... ({len(field_value)} total)"
                elif isinstance(field_value, float):
                    # Format frequencies/scores nicely
                    if field_value < 0.0001:
                        value_str = f"{field_value:.2e}"
                    elif field_value < 0.01:
                        value_str = f"{field_value:.6f}"
                    else:
                        value_str = f"{field_value:.4f}"
                elif isinstance(field_value, bool):
                    value_str = "Yes" if field_value else "No"
                else:
                    value_str = str(field_value)
                    if len(value_str) > 150:
                        value_str = value_str[:150] + "..."

                formatted_sections.append(f"  • {field_name}: {value_str}")

        formatted_data = "\n".join(formatted_sections)

        if self.debug:
            try:
                reduction_pct = (filtered_fields / max(1, (kept_fields + filtered_fields)) * 100)
            except Exception:
                reduction_pct = 0.0
            print(f"🔧 DEBUG: Formatted output length: {len(formatted_data)} characters")
            print(f"🔧 DEBUG: Reduction vs unfiltered: ~{reduction_pct:.1f}% fewer fields")

        return formatted_data

    def create_clinical_interpretation_prompt(self, data: str, source_name: str) -> str:
        """Create prompt with explicit interpretation rules and citation requirements"""
        is_mistral = "mistral" in self.model_name.lower()

        interpretation_rules = """
CRITICAL INTERPRETATION RULES:

1. **Prediction Score Interpretation:**
   - SIFT: 'T' = Tolerated (BENIGN), 'D' = Deleterious (DAMAGING)
   - SIFT score: <0.05 = deleterious, ≥ 0.05 = tolerated
   - PolyPhen: 'B' = Benign, 'P' = Possibly damaging, 'D' = Probably damaging
   - PolyPhen score: >0.85 = probably damaging, 0.45-0.85 = possibly damaging, <0.45 = benign
   - CADD: Higher scores = more deleterious (>20 is damaging, >30 is highly damaging)

2. **ClinVar Data Handling:**
   - If clinvar_rcv_count > 0, there ARE ClinVar records even if clinvar_conditions is null
   - Say: "ClinVar reports N RCV accessions [clinvar_rcv_count] but specific disease associations are not available in this field [clinvar_conditions]"
   - Never say "no ClinVar data" if clinvar_rcv_count exists

3. **Population Frequency Citation:**
   - ALWAYS cite the actual numeric value, not just "low" or "high"
   - Format: "South Asian AF = 0.000123 [gnomad_exome_af_af_sas]"
   - If a frequency field exists but is null, say "not reported [field_name]"

4. **Conflicting Evidence:**
   - If SIFT says Tolerated but PolyPhen says Damaging, acknowledge the conflict explicitly
   - Example: "SIFT predicts tolerated [sift_pred=T] but PolyPhen predicts possibly damaging [polyphen_pred=P], indicating conflicting evidence"

5. **Missing Data vs Null Fields:**
   - Distinguish between "field doesn't exist" vs "field exists but is null"
   - If a field is null, say: "not available in this annotation [field_name]"
   - Don't claim data is "missing" if the field simply wasn't queried

6. **HGVS Notation:**
   - Check both VEP and MyVariant sources for HGVS
   - VEP may have hgvsc/hgvsp in transcript-specific fields
   - MyVariant may have them at root level or in nested structures
"""

        if is_mistral:
            prompt = f"""<s>[INST] You are a clinical geneticist analyzing {source_name} data. Follow interpretation rules strictly.

{interpretation_rules}

{source_name.upper()} DATA:
{data}

PROVIDE CLINICAL ASSESSMENT:

1. **Pathogenicity Assessment** (2-3 sentences):
   - Cite EXACT numeric values: "CADD score of 24.3 [cadd_phred] suggests..."
   - Correctly interpret prediction codes (SIFT T=Tolerated, D=Damaging)
   - If ClinVar RCV count exists, mention it even if conditions are null

2. **Key Concerns** (bullet points with citations):
   - Population frequency: cite actual AF values
   - Conflicting predictions: explicitly state which tools disagree
   - Missing vs null fields: distinguish clearly

3. **Functional Impact** (2-3 sentences):
   - Amino acid change if available
   - Consequence type [most_severe_consequence or consequence_terms]
   - Which predictions support the assessment

4. **Clinical Recommendation** (1-2 sentences):
   - Classification suggestion based on evidence
   - Acknowledge uncertainty where appropriate

**CITATION FORMAT:**
Every claim MUST cite the source field: "SIFT score of 0.42 [sift_score] predicts tolerated [sift_pred=T]"

[/INST]"""
        else:
            prompt = f"""You are a clinical geneticist analyzing {source_name} data.

{interpretation_rules}

{source_name.upper()} DATA:
{data}

Provide clinical assessment with proper interpretation and citations.
"""

        return prompt

    def extract_response(self, response: str, prompt: str, source_name: str) -> str:
        """Extract the actual response from model output"""
        if self.debug:
            print(f"🔧 DEBUG: Response length: {len(response)}")

        # Method 1: Look for [/INST] (Mistral)
        if "[/INST]" in response:
            parts = response.split("[/INST]")
            if len(parts) > 1:
                return parts[1].strip()

        # Method 2: Remove prompt
        if len(response) > len(prompt):
            return response[len(prompt):].strip()

        # Method 3: Full response
        return response.strip()

    def generate_individual_summary(self, df: pd.DataFrame, source_name: str) -> str:
        """Generate clinical interpretation with citations"""
        if df.empty:
            return f"No {source_name} data available."

        if self.debug:
            print(f"\n🔧 DEBUG: Starting {source_name} CLINICAL INTERPRETATION")

        try:
            # Format FILTERED data
            formatted_data = self.format_dataframe_for_prompt(df, source_name)

            # Create INTERPRETATION prompt (not reformatting prompt)
            prompt = self.create_clinical_interpretation_prompt(formatted_data, source_name)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
                padding=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            if self.debug:
                print(f"🔧 DEBUG: Input tokens: {inputs['input_ids'].shape[1]}")

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    min_new_tokens=150,  # Force longer, more detailed response
                    temperature=0.3,     # Lower = more factual
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.15,  # Discourage repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1
                )

            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract summary
            summary = self.extract_response(response, prompt, source_name)

            # Add disclaimer if not already present
            if "disclaimer" not in summary.lower():
                summary += "\n\n⚠️ **DISCLAIMER**: This is an automated AI interpretation for research and educational purposes only. This analysis does not constitute medical advice. All clinical decisions must be made by qualified healthcare professionals after comprehensive review of the patient's complete medical history, family history, and clinical presentation."

            if self.debug:
                print(f"🔧 DEBUG: Final summary length: {len(summary)} characters")

            return summary if len(summary) > 0 else self._emergency_fallback(df, source_name)

        except Exception as e:
            if self.debug:
                print(f"❌ DEBUG: Error in {source_name} interpretation: {e}")
            return self._emergency_fallback(df, source_name)

    def _emergency_fallback(self, df: pd.DataFrame, source_name: str) -> str:
        """Emergency fallback with basic interpretation"""
        if df.empty:
            return f"No {source_name} data available."

        data_dict = df.iloc[0].to_dict()

        # Apply same filtering
        filtered_data = {}
        for field_name, field_value in data_dict.items():
            if pd.notna(field_value) and field_value != '' and field_value != []:
                if POPULATION_FILTER.should_include_field(field_name):
                    filtered_data[field_name] = field_value

        if self.debug:
            print(f"🛟 DEBUG: Emergency fallback for {source_name} with {len(filtered_data)} filtered fields")

        output = f"**{source_name.upper()} DATA (BASIC SUMMARY)**\n"
        output += "="*60 + "\n"

        # Categorize
        categorized = {}
        for field_name, field_value in filtered_data.items():
            category = POPULATION_FILTER.get_category_for_field(field_name)
            if category not in categorized:
                categorized[category] = {}
            categorized[category][field_name] = field_value

        # Output key categories only
        priority_categories = [
            "VARIANT_IDENTIFICATION", "CLINICAL_SIGNIFICANCE",
            "FUNCTIONAL_PREDICTIONS", "POPULATION_FREQUENCIES"
        ]

        for category in priority_categories:
            if category not in categorized:
                continue

            output += f"\n**{category.replace('_', ' ')}**\n"
            for key, value in sorted(categorized[category].items())[:8]:  # Limit to 8 fields
                if isinstance(value, list):
                    value_str = ", ".join(str(item) for item in value[:5])
                    if len(value) > 5:
                        value_str += f" ... ({len(value)} total)"
                else:
                    value_str = str(value)

                output += f"• {key}: {value_str}\n"

        output += "\n⚠️ **DISCLAIMER**: Automated interpretation only. Clinical decisions require professional review.\n"
        output += "="*60
        return output

    def generate_combined_summary(self, vep_df: pd.DataFrame, myvariant_df: pd.DataFrame) -> Dict[str, str]:
        """Generate clinical interpretations for both data sources"""
        if self.debug:
            print("\n🚀 STARTING CLINICAL INTERPRETATION GENERATION")
            print("="*60)

        summaries = {}

        # VEP Interpretation
        if self.debug:
            print("\n🧬 PROCESSING VEP DATA FOR CLINICAL INSIGHTS")
        summaries['vep'] = self.generate_individual_summary(vep_df, "VEP")

        # MyVariant Interpretation
        if self.debug:
            print("\n🧬 PROCESSING MYVARIANT DATA FOR CLINICAL INSIGHTS")
        summaries['clinvar'] = self.generate_individual_summary(myvariant_df, "MyVariant")

        if self.debug:
            print(f"\n✅ Clinical interpretations generated:")
            print(f"   VEP: {len(summaries['vep'])} chars")
            print(f"   MyVariant: {len(summaries['clinvar'])} chars")

        return summaries