"""
Stage 4: Specialty Verification
Verifies whether queries match their assigned medical specialties.
"""
import os
import json
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

from base_processor import BaseProcessor, FileManager, RetryProcessor, DataSplitter
from config import ConfigManager
from logging_utils import PipelineLogger, get_logger
from error_handling import handle_errors, validate_inputs
from azure_utils import AzureModelManager


class SpecialtyVerificationConfig:
    """Configuration for specialty verification."""
    
    VERIFICATION_PROMPT = """You are a certified medical coder who assigns ICD-10 codes.

    **Goal:**  
    Given (1) a medical search query, (2) a user-supplied list of **ICD-10 code: description** pairs, and (3) a reference medical **specialty_subspecialty**, identify whether the reference **specialty_subspecialty** is **relevant** or **non-relevant** to the medical query and the ICD-10 code-description list. The result should be **non-relevant** if the reference **specialty_subspecialty** is either too generic for the query/ICD-10 list's intent or clearly unrelated to that query and code list.
    
    **How to decide relevance:**  
    - **Understand the query's clinical intent:** Determine what condition, symptom, or scenario the query is describing.  
    - **Consider the ICD-10 code list context:** The ICD-10 codes and descriptions are chosen to be consistent with each other and with the query, representing a specific clinical scenario. Use this combined context to inform your decision.  
    - **Match with the specialty_subspecialty:** Evaluate whether a practitioner of the reference **specialty_subspecialty** typically addresses the query's scenario:  
      - The specialty_subspecialty pair may consist of a broad specialty and a more focused subspecialty. Emphasize the general **specialty** domain. If the scenario falls under that general field (even if not an exact subspecialty match), it can be considered relevant.  
      - If the scenario (query + codes) **does not fall under** the domain of the reference specialty_subspecialty — for example, the specialty_subspecialty is overly broad/vague for this specific case, or it pertains to a different field of medicine — then label it **non-relevant**.  
      - If the scenario **does** fall under the clinical domain of that specialty_subspecialty (i.e. a provider of that type would reasonably handle such cases), then label it **relevant**.  
    - **Multiple possible specialties:** There may be cases where the query and codes could belong to more than one specialty. You are **only** checking the given reference specialty_subspecialty. If the given specialty_subspecialty is one appropriate choice for this scenario, mark it **relevant** (even if other specialties could also be involved).  
    - **If unsure:** If you cannot confidently determine relevance from the information provided, label the result as `CANNOT_DECIDE`.
    
    **Response format (strict):**  
    Return **only** a single label as the answer: `relevant`, `non-relevant`, or `CANNOT_DECIDE` (use `CANNOT_DECIDE` only if you truly cannot decide). Do **not** include explanations, reasoning, or any additional text.
    
    **Inputs (to be inserted at runtime):**  
    medical_query: *{medical_query}*  
    icd_code_description_list: *{icd_code_description_list}*  
    medical_specialty_subspecialty: *{medical_specialty_subspecialty}*  
    
    **Few-shot guidance (examples):**
    
    Example 1:  
    medical_query: **aging and decreased independence and mobility**  
    icd_code_description_list: **['Z74.3 : Need for continuous supervision', 'Z73.89 : Other problems related to life management difficulty', 'Z73.6 : Limitation of activities due to disability', 'Z60.0 : Phase of life problem', 'Z74.2 : Need for assistance at home and no other household member able to render care', 'Z74.1 : Need for assistance with personal care', 'Z74.09 : Other reduced mobility', 'R54 : Senile debility', 'Z91.81 : History of falling']**  
    medical_specialty_subspecialty: **adult companion_adult companion**  
    Expected output: **relevant**
    
    Example 2:  
    medical_query: **acupuncture for headaches**  
    icd_code_description_list: **['R51.9 : Headache, unspecified', 'G44.209 : Tension-type headache, unspecified, not intractable', 'G43.009 : Migraine without aura NOS', 'G44.89 : Other headache syndrome', 'G43.909 : Migraine NOS', 'G43.709 : Chronic migraine without aura NOS']**  
    medical_specialty_subspecialty: **anesthesiology_addiction medicine**  
    Expected output: **non-relevant**
    
    Example 3:  
    medical_query: **specialty biologic and injectable therapies in healthcare**  
    icd_code_description_list: **['T88.59XA : Other complications of anesthesia, initial encounter', 'T41.1X5A : Adverse effect of intravenous anesthetics, initial encounter', 'T88.7XXA : Unspecified adverse effect of drug or medicament, initial encounter']**  
    medical_specialty_subspecialty: **cliniccenter_student health**  
    Expected output: **non-relevant**
    
    Example 4:  
    medical_query: **itchy scalp after workplace exposure**  
    icd_code_description_list: **['L23.9 : Allergic contact dermatitis, unspecified cause', 'L28.0 : Circumscribed neurodermatitis', 'L25.9 : Unspecified contact dermatitis, unspecified cause', 'L23.8 : Allergic contact dermatitis due to other agents', 'L23.5 : Allergic contact dermatitis due to plastic', 'L29.8 : Other pruritus', 'R21 : Rash and other nonspecific skin eruption', 'L50.9 : Urticaria, unspecified', 'L24.9 : Irritant contact dermatitis, unspecified cause', 'L27.2 : Dermatitis due to ingested food', 'L24.0 : Irritant contact dermatitis due to detergents']**  
    medical_specialty_subspecialty: **cardiologist**  
    Expected output: **non-relevant**
    
    Example 5:  
    medical_query: **acne worsening at job**  
    icd_code_description_list: **['L70.1 : Acne conglobata', 'L21.9 : Seborrheic dermatitis, unspecified', 'L30.9 : Eczema NOS', 'L25.9 : Unspecified contact dermatitis, unspecified cause', 'L70.0 : Acne vulgaris', 'L23.5 : Allergic contact dermatitis due to plastic', 'L71.9 : Rosacea, unspecified', 'L24.9 : Irritant contact dermatitis, unspecified cause', 'L70.8 : Other acne', 'L70.9 : Acne, unspecified', 'L24.0 : Irritant contact dermatitis due to detergents', 'L71.0 : Perioral dermatitis']**  
    medical_specialty_subspecialty: **dermatopathology_occupational medicine**  
    Expected output: **relevant**

    Example 6:  
    medical_query: **how does diabetes affect brain tumour**  
    icd_code_description_list: **['L70.1 : Acne conglobata', 'L21.9 : Seborrheic dermatitis, unspecified', 'L30.9 : Eczema NOS', 'L25.9 : Unspecified contact dermatitis, unspecified cause', 'L70.0 : Acne vulgaris', 'L23.5 : Allergic contact dermatitis due to plastic', 'L71.9 : Rosacea, unspecified', 'L24.9 : Irritant contact dermatitis, unspecified cause', 'L70.8 : Other acne', 'L70.9 : Acne, unspecified', 'L24.0 : Irritant contact dermatitis due to detergents', 'L71.0 : Perioral dermatitis']**  
    medical_specialty_subspecialty: **neurology**  
    Expected output: **CANNOT_DECIDE**
    
    **Remember:** Provide **only** the label (`relevant`, `non-relevant`, or `CANNOT_DECIDE`) as the answer. Do not add any explanation or extra text.
    """


class ICDDataManager:
    """Manages ICD reference data for specialty verification."""
    
    def __init__(self, file_manager: FileManager, logger: PipelineLogger):
        self.file_manager = file_manager
        self.logger = logger
    
    def load_icd_reference(self, icd_reference_file: str) -> Dict[str, str]:
        """Load ICD reference data."""
        try:
            dataset_icd = pd.read_csv(icd_reference_file).iloc[:, 1:]
            dataset_icd = dataset_icd.drop_duplicates()
            dataset_icd = dataset_icd.iloc[:, 13:15]
            dataset_icd.columns = ['ICD_Codes', 'Description']
            dataset_icd['ICD_Codes'] = dataset_icd['ICD_Codes'].apply(lambda x: x.strip())
            dataset_icd['Description'] = dataset_icd['Description'].apply(lambda x: x.strip())
            dataset_icd = dataset_icd.drop_duplicates(subset=['ICD_Codes'], keep='first')

            icd_reference_lookup = {}
            for index, row in dataset_icd.iterrows():
                icd_reference_lookup[row.ICD_Codes] = row.Description

            self.logger.info(f"Loaded {len(icd_reference_lookup)} ICD codes from reference file")
            return icd_reference_lookup
            
        except Exception as e:
            self.logger.error(f"Failed to load ICD reference file: {e}")
            raise


class FilteredDataLoader:
    """Loads filtered ICD code data from previous processing stages."""
    
    def __init__(self, file_manager: FileManager, logger: PipelineLogger):
        self.file_manager = file_manager
        self.logger = logger
    
    def load_filtered_datasets(self, file_path: str) -> Dict[str, Dict[str, List[str]]]:
        """Load filtered specialty query datasets from individual files."""
        all_files = self.file_manager.list_files(file_path, '.json')
        
        specialty_query_codes_dict = {}
        
        for file_path_item in all_files:
            try:
                data = self.file_manager.load_json(file_path_item)
                specialty = list(data.keys())[0]
                query_codes_dict = list(data.values())[0]
                
                # Filter out queries with no ICD codes
                filtered_query_codes = {}
                for query, codes in query_codes_dict.items():
                    if codes and len(codes) > 0:
                        filtered_query_codes[query] = codes
                
                if filtered_query_codes:
                    