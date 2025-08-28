"""
Stage 3: Unified ICD Code Generation and Verification
Handles both ICD code generation and verification in a single pipeline.
"""
import os
import json
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from base_processor import BaseProcessor, FileManager, RetryProcessor, DataSplitter
from config import ConfigManager
from logging_utils import PipelineLogger, get_logger
from error_handling import handle_errors, validate_inputs, validate_positive_integer
from azure_utils import AzureModelManager


class ICDGenerationConfig:
    """Configuration for ICD code generation."""
    
    GENERATION_PROMPT = """You are a certified medical coder specializing in ICD-10 code assignment for search queries.
    Your task is to generate correct and relevant ICD-10 codes based on a medical query and medical_specialty provided by the user. 
    
    Please note that the medical_specialty provided by the user contains a specialty and a subspecialty as follows : specialty_subspecialty.
    For example if the specialty is internal medicine and subscpecialty is endocrinology then the user input would be internal medicine_endocrinology
    
    Please develop an understanding of the intent of the user provided medical search query with respect to the medical_specialty
    and adhere to the following guidelines to generate the ICD codes:
    
    Strict Guideline:  
    - Return only valid ICD-10 codes separated by comma , without additional text.
    - Generate upto 10 ICD-10 codes that are most relevant to the medical_specialty for the identified medical terms
    - Do not suggest too generic or very broad range of ICD codes which can be mapped to multiple medical_specialties for the given query, instead suggest codes which are highly specific to the query under the given medical_specialty
    - If there are no ICD codes for the medical query, return "[]" to indicate an empty list.
    
    Query Analysis Rules:
    1. If the query contains ONLY names or locations without medical terms example santa monica, bellevue, return "[]".
    2. If the query contains the medical_specialty itself (e.g. "cardiologist" for Cardiology) generate top relevant ICD-10 codes for the medical_specialty
    3. If the query contain conditions/procedures related to the medical_specialty, generate top relevant ICD-10 codes for the searhch query for that medical_specialty
    4. For any query mentioning a medical profession role ( e.g. "physiotherapist", "cardiologist","oncologist", "dentist" etc ), extract the specialty
    and generate common ICD-10 codes for conditions typically treated by that specialty. 
       Exaple : sanata monda orthopedics group. Instruction : Extract orthopedics from the query and generate ICD-10 codes for that
    4. Always prioritize specific medical conditions mentioned in the query (e.g. "knee pain","back spasm") over generic specialty terms.
    5. If the query contains both a specialty term AND a specific condition, prioritize codes for the specific condition.
    6. Analyze the search query and extract relevant tokens from the search query which can be mapped to ICD-10 codes, after extraction return the relevant ICD-10 codes

    IMPORTANT:
    PLEASE TAKE YOUR TIME IN UNDERSTANDING THE MEDICAL QUERY WITH RESPECT TO Medical specialty. Your response must contain
    ONLY upto 10 relevant ICD-10 codes separated by commas, or "[]" if no codes apply. Please make sure that the codes you suggest are limited to the
    search query and medical_specialty, Do not suggest too generic or very broad range of ICD codes which can be mapped to multiple medical_specialties for the given query
    
    Do not include any explanations, headers, or additional text in your response.
    PLEASE DO NOT DEVIATE FROM THE ABOVE ASSUMPTION. 
                        
    Task:  
    Identify the correct ICD codes for the following medical query.  
    
    Format Instructions:
    {format_instructions}
    
    medical_query: {medical_query}
    medical_specialty: {medical_specialty}
    """

    VERIFICATION_PROMPT = """You are a certified medical coder who assigns ICD-10 codes.

    Goal  
    Given (1) a medical search query, (2) a medical **specialty_subspecialty**, and (3) a user-supplied list of **ICD-10 code : description** pairs, identify which codes are **non-relevant**—i.e., either too generic for the query's intent or unrelated to the stated specialty_subspecialty.
    
    How to decide relevance  
    • Understand the clinical intent expressed in the query.  
    • Align that intent with the clinical scope of the specialty_subspecialty.  
    • Examine each ICD-10 description in the list.  
      – If it does **not** match both the query intent **and** the specialty_subspecialty with reasonable clinical specificity, mark it non-relevant.  
      – Otherwise, treat it as relevant (do *not* output it).
    
    Response format (strict)  
    Return **only** the non-relevant ICD-10 codes, separated by commas.  
    Example: `A00.1,Y38.2`  
    If every code is relevant, return `[]` (just the two bracket characters).  
    Do **not** include explanations, headings, or extra text.
    
    Inputs (to be injected at runtime)  
    medical_query: {medical_query}
    medical_specialty_subspecialty: {medical_specialty_subspecialty}
    icd_code_description_list: {icd_code_description_list}
    
    Few-shot guidance 
    Example 1
    medical_query: swelling
    medical_specialty_subspecialty: acupuncturist_acupuncturist
    icd_code_description_list: ["R60.9: Fluid retention NOS", "Y38.2: Terrorism involving other explosions", "A00.1: Cholera due to Vibrio cholerae"]
    Expected output: Y38.2,A00.1
    
    Example 2
    medical_query: eye socket tumour
    medical_specialty_subspecialty: cliniccenter_oral_and_maxillofacial_surgery
    icd_code_description_list: ["C41.0: Malignant neoplasm of skull bones", "D3A.01: Benign carcinoid tumour of small intestine"]
    Expected output: D3A.01
    
    Remember: output **only** the comma-separated ICD-10 codes (or `[]`). Do not add any other text.
    """


class ICDDataManager:
    """Manages ICD reference data."""
    
    def __init__(self, file_manager: FileManager, logger: PipelineLogger):
        self.file_manager = file_manager
        self.logger = logger
        self.icd_reference_lookup = {}
    
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

            self.icd_reference_lookup = icd_reference_lookup
            self.logger.info(f"Loaded {len(icd_reference_lookup)} ICD codes from reference file")
            return icd_reference_lookup
            
        except Exception as e:
            self.logger.error(f"Failed to load ICD reference file: {e}")
            raise


class ICDGenerator:
    """Generates ICD codes using Azure OpenAI."""
    
    def __init__(self, azure_manager: AzureModelManager, logger: PipelineLogger):
        self.azure_manager = azure_manager
        self.logger = logger
        self.output_parser = CommaSeparatedListOutputParser()
        self.prompt_template = PromptTemplate.from_template(ICDGenerationConfig.GENERATION_PROMPT)
    
    @handle_errors(continue_on_error=True)
    def generate_icd_codes(self, model, medical_specialty: str, medical_query: str) -> Optional[List[str]]:
        """Generate ICD codes for a medical query."""
        chain = self.prompt_template | model | self.output_parser
        
        result = chain.invoke(input={
            "medical_specialty": medical_specialty,
            "medical_query": medical_query,
            "format_instructions": self.output_parser.get_format_instructions()
        })
        
        return result if result else []


class ICDVerifier:
    """Verifies and filters ICD codes using Azure OpenAI."""
    
    def __init__(self, azure_manager: AzureModelManager, logger: PipelineLogger):
        self.azure_manager = azure_manager
        self.logger = logger
        self.output_parser = CommaSeparatedListOutputParser()
        self.prompt_template = PromptTemplate.from_template(ICDGenerationConfig.VERIFICATION_PROMPT)
    
    @handle_errors(continue_on_error=True)
    def filter_irrelevant_codes(self, model, medical_specialty: str, medical_query: str, 
                               icd_code_description_list: List[str]) -> Optional[List[str]]:
        """Filter out irrelevant ICD codes."""
        chain = self.prompt_template | model | self.output_parser
        
        result = chain.invoke(input={
            "medical_specialty_subspecialty": medical_specialty,
            "medical_query": medical_query,
            "icd_code_description_list": icd_code_description_list,
            "format_instructions": self.output_parser.get_format_instructions()
        })
        
        return result if result else []


class ICDProcessor(BaseProcessor):
    """Unified processor for ICD generation and verification."""
    
    def __init__(self, config: ConfigManager, logger: PipelineLogger):
        super().__init__(config, logger, "STAGE3_ICD_PROCESSING")
        self.file_manager = FileManager(logger)
        self.data_splitter = DataSplitter(logger, self.file_manager)
        self.icd_data_manager = ICDDataManager(self.file_manager, logger)
        self.icd_generator = ICDGenerator(self.azure_manager, logger)
        self.icd_verifier = ICDVerifier(self.azure_manager, logger)
        self.retry_processor = RetryProcessor(logger, config.processing.retry_attempts)
    
    def load_medical_specialty_datasets(self, chunk_index: Optional[int] = None) -> Dict[str, List[str]]:
        """Load medical specialty datasets."""
        stage_paths = self.config.get_stage_paths('stage3')
        input_path = stage_paths['input_path']
        
        all_files = self.file_manager.list_files(input_path, '.json')
        
        if chunk_index is not None:
            if chunk_index >= len(all_files):
                raise ValueError(f"Chunk index {chunk_index} out of range (0-{len(all_files)-1})")
            all_files = [all_files[chunk_index]]
        
        combined_data = {}
        for file_path in all_files:
            data = self.file_manager.load_json(file_path)
            combined_data.update(data)
        
        self.logger.info(f"Loaded {len(combined_data)} specialties from {len(all_files)} files")
        return combined_data
    
    def generate_icd_codes_stage(self, medical_specialty_dataset: Dict[str, List[str]], 
                                models: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate ICD codes for medical queries."""
        from tqdm import tqdm
        
        stage_paths = self.config.get_stage_paths('stage3')
        gpt_4o_path = stage_paths['gpt_4o_results']
        gpt_41_path = stage_paths['gpt_41_results']
        
        self.file_manager.ensure_directory(gpt_4o_path)
        self.file_manager.ensure_directory(gpt_41_path)
        
        # Get already processed specialties
        processed_gpt4o = self.file_manager.get_processed_files(gpt_4o_path)
        processed_gpt41 = self.file_manager.get_processed_files(gpt_41_path)
        processed_specialties = set(processed_gpt4o).intersection(set(processed_gpt41))
        
        all_specialties = list(medical_specialty_dataset.keys())
        remaining_specialties = [s for s in all_specialties if s not in processed_specialties]
        
        self.logger.info(f"Processing {len(remaining_specialties)} remaining specialties for ICD generation")
        
        retry_items = defaultdict(list)
        
        for specialty in tqdm(remaining_specialties, desc="Generating ICD codes"):
            self.logger.info(f'Processing Specialty: {specialty}')
            
            medical_queries = medical_specialty_dataset[specialty]
            
            query_set_gpt_41 = {}
            query_set_gpt_4o = {}
            
            for query in tqdm(medical_queries, desc=f"Processing {specialty}", leave=False):
                try:
                    # Generate with GPT-4.1
                    icd_codes_gpt_41 = self.retry_processor.execute_with_retry(
                        self.icd_generator.generate_icd_codes,
                        f"{specialty}_{query[:50]}_gpt41",
                        models['gpt-4.1'], specialty, query
                    )
                    
                    # Generate with GPT-4o
                    icd_codes_gpt_4o = self.retry_processor.execute_with_retry(
                        self.icd_generator.generate_icd_codes,
                        f"{specialty}_{query[:50]}_gpt4o",
                        models['gpt-4o'], specialty, query
                    )
                    
                    if icd_codes_gpt_41:
                        query_set_gpt_41[query] = icd_codes_gpt_41
                    if icd_codes_gpt_4o:
                        query_set_gpt_4o[query] = icd_codes_gpt_4o
                        
                except Exception as e:
                    self.logger.error(f"Failed to generate ICD codes for {specialty}/{query}: {e}")
                    retry_items[specialty].append(query)
            
            # Save results
            if query_set_gpt_41:
                specialty_data_41 = {specialty: query_set_gpt_41}
                output_file_41 = Path(gpt_41_path) / f'{specialty}.json'
                self.file_manager.save_json(specialty_data_41, str(output_file_41))
            
            if query_set_gpt_4o:
                specialty_data_4o = {specialty: query_set_gpt_4o}
                output_file_4o = Path(gpt_4o_path) / f'{specialty}.json'
                self.file_manager.save_json(specialty_data_4o, str(output_file_4o))
            
            self.logger.info(f'Completed Specialty: {specialty}')
            time.sleep(0.1)
        
        return dict(retry_items)
    
    def combine_gpt_results(self, icd_reference_lookup: Dict[str, str]) -> Tuple[Dict, Dict]:
        """Combine GPT-4o and GPT-4.1 results."""
        stage_paths = self.config.get_stage_paths('stage3')
        gpt_4o_path = stage_paths['gpt_4o_results']
        gpt_41_path = stage_paths['gpt_41_results']
        
        all_files_gpt4o = self.file_manager.list_files(gpt_4o_path, '.json')
        all_files_gpt41 = self.file_manager.list_files(gpt_41_path, '.json')
        
        if len(all_files_gpt4o) != len(all_files_gpt41):
            self.logger.warning(f"Mismatch in file counts: GPT-4o={len(all_files_gpt4o)}, GPT-4.1={len(all_files_gpt41)}")
        
        specialty_query_dict = {}
        specialty_query_code_description_dict = {}
        problematic_specialties = []
        problematic_queries = []
        
        for i, file_path_4o in enumerate(all_files_gpt4o):
            if i >= len(all_files_gpt41):
                break
                
            file_path_41 = all_files_gpt41[i]
            specialty = Path(file_path_4o).stem
            
            try:
                data_gpt4o = self.file_manager.load_json(file_path_4o)
                data_gpt41 = self.file_manager.load_json(file_path_41)
                
                queries = list(data_gpt4o.get(specialty, {}).keys())
                
                query_code_dict = {}
                query_code_description_dict = {}
                
                for query in queries:
                    try:
                        codes_gpt4o = data_gpt4o.get(specialty, {}).get(query, [])
                        codes_gpt41 = data_gpt41.get(specialty, {}).get(query, [])
                        
                        # Combine and deduplicate codes
                        combined_codes = list(set(codes_gpt4o + codes_gpt41))
                        
                        # Clean codes (remove trailing zeros)
                        cleaned_codes = []
                        for code in combined_codes:
                            if len(code) >= 2 and code.endswith('00'):
                                cleaned_codes.append(code[:-1])
                            else:
                                cleaned_codes.append(code)
                        
                        # Validate codes against reference
                        valid_codes = []
                        code_descriptions = []
                        
                        for code in cleaned_codes:
                            if code in icd_reference_lookup:
                                valid_codes.append(code)
                                code_descriptions.append(f"{code} : {icd_reference_lookup[code]}")
                        
                        if valid_codes:
                            query_code_dict[query] = valid_codes
                            query_code_description_dict[query] = code_descriptions
                        
                    except Exception as e:
                        problematic_queries.append(f"{specialty} : {query}")
                        self.logger.warning(f"Error processing query {query} in {specialty}: {e}")
                
                if query_code_dict:
                    specialty_query_dict[specialty] = query_code_dict
                    specialty_query_code_description_dict[specialty] = query_code_description_dict
                
            except Exception as e:
                problematic_specialties.append(specialty)
                self.logger.error(f"Error processing specialty {specialty}: {e}")
        
        # Save combined results
        filtered_path = stage_paths['filtered_path']
        self.file_manager.ensure_directory(filtered_path)
        
        combined_codes_path = Path(filtered_path) / 'specialty_query_dict.json'
        combined_descriptions_path = Path(filtered_path) / 'specialty_query_code_description_dict.json'
        
        self.file_manager.save_json(specialty_query_dict, str(combined_codes_path))
        self.file_manager.save_json(specialty_query_code_description_dict, str(combined_descriptions_path))
        
        self.logger.info(f"Combined results for {len(specialty_query_dict)} specialties")
        if problematic_specialties:
            self.logger.warning(f"Problematic specialties: {len(problematic_specialties)}")
        if problematic_queries:
            self.logger.warning(f"Problematic queries: {len(problematic_queries)}")
        
        return specialty_query_dict, specialty_query_code_description_dict
    
    def verify_icd_codes_stage(self, specialty_query_dict: Dict, specialty_query_code_description_dict: Dict,
                              model, chunk_index: Optional[int] = None) -> Dict[str, List[str]]:
        """Verify and filter ICD codes."""
        from tqdm import tqdm
        
        # Split data if chunk processing is requested
        if chunk_index is not None:
            all_specialties = list(specialty_query_dict.keys())
            chunk_size = len(all_specialties) // self.config.processing.num_chunks + 1
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_specialties))
            
            specialties_to_process = all_specialties[start_idx:end_idx]
            self.logger.info(f"Processing chunk {chunk_index}: specialties {start_idx}-{end_idx-1}")
        else:
            specialties_to_process = list(specialty_query_dict.keys())
        
        stage_paths = self.config.get_stage_paths('stage3')
        output_path = Path(stage_paths['filtered_path']) / 'filtered_icd_codes'
        self.file_manager.ensure_directory(str(output_path))
        
        # Get already processed specialties
        processed_specialties = self.file_manager.get_processed_files(str(output_path))
        remaining_specialties = [s for s in specialties_to_process if s not in processed_specialties]
        
        self.logger.info(f"Verifying {len(remaining_specialties)} remaining specialties")
        
        retry_items = defaultdict(list)
        
        for specialty in tqdm(remaining_specialties, desc="Verifying ICD codes"):
            self.logger.info(f'Processing Specialty: {specialty}')
            
            query_code_dict = specialty_query_dict.get(specialty, {})
            query_code_description_dict = specialty_query_code_description_dict.get(specialty, {})
            
            filtered_query_code_dict = {}
            
            for query, code_descriptions in tqdm(query_code_description_dict.items(), 
                                               desc=f"Verifying {specialty}", leave=False):
                try:
                    # Get irrelevant codes
                    irrelevant_codes = self.retry_processor.execute_with_retry(
                        self.icd_verifier.filter_irrelevant_codes,
                        f"{specialty}_{query[:50]}",
                        model, specialty, query, code_descriptions
                    )
                    
                    # Filter out irrelevant codes
                    original_codes = query_code_dict.get(query, [])
                    if irrelevant_codes:
                        filtered_codes = [code for code in original_codes if code not in irrelevant_codes]
                    else:
                        filtered_codes = original_codes
                    
                    if filtered_codes:
                        filtered_query_code_dict[query] = filtered_codes
                    
                except Exception as e:
                    self.logger.error(f"Error verifying codes for {specialty}/{query}: {e}")
                    retry_items[specialty].append(query)
            
            # Save filtered results
            if filtered_query_code_dict:
                specialty_data = {specialty: filtered_query_code_dict}
                output_file = output_path / f'{specialty}.json'
                self.file_manager.save_json(specialty_data, str(output_file))
            
            self.logger.info(f'Completed Specialty: {specialty}')
            time.sleep(0.1)
        
        return dict(retry_items)
    
    @validate_inputs(
        chunk_index=lambda x: x is None or (isinstance(x, int) and x >= 0)
    )
    def process(self,
                mode: str = "all",
                chunk_index: Optional[int] = None,
                combine_results: bool = True,
                create_splits: bool = False,
                load_splits: bool = False) -> Dict[str, Any]:
        """
        Process ICD generation and verification.
        
        Args:
            mode: Processing mode ('generate', 'combine', 'verify', 'all')
            chunk_index: Optional chunk index for distributed processing
            combine_results: Whether to combine GPT results
            create_splits: Whether to create data splits
            load_splits: Whether to load existing splits
        
        Returns:
            Dictionary containing processing results and retry items
        """
        results = {}
        
        # Load ICD reference data
        icd_reference_lookup = self.icd_data_manager.load_icd_reference(
            self.config.paths.icd_reference_file
        )
        
        # Initialize models if needed
        models = {}
        if mode in ['generate', 'all']:
            models['gpt-4o'] = self.azure_manager.initialize_model('gpt-4o')
            models['gpt-4.1'] = self.azure_manager.initialize_model('gpt-4.1')
        elif mode in ['verify', 'all']:
            models['gpt-4.1'] = self.azure_manager.initialize_model('gpt-4.1')
        
        # Step 1: Generate ICD codes
        if mode in ['generate', 'all']:
            self.logger.info("=== Starting ICD Code Generation ===")
            medical_specialty_dataset = self.load_medical_specialty_datasets(chunk_index)
            retry_generate = self.generate_icd_codes_stage(medical_specialty_dataset, models)
            results['generate_retry'] = retry_generate
        
        # Step 2: Combine results
        if mode in ['combine', 'all'] and combine_results:
            self.logger.info("=== Starting Result Combination ===")
            specialty_query_dict, specialty_query_code_description_dict = self.combine_gpt_results(
                icd_reference_lookup
            )
            results['combined_specialties'] = len(specialty_query_dict)
        else:
            # Load existing combined results
            stage_paths = self.config.get_stage_paths('stage3')
            filtered_path = stage_paths['filtered_path']
            
            combined_codes_path = Path(filtered_path) / 'specialty_query_dict.json'
            combined_descriptions_path = Path(filtered_path) / 'specialty_query_code_description_dict.json'
            
            specialty_query_dict = self.file_manager.load_json(str(combined_codes_path))
            specialty_query_code_description_dict = self.file_manager.load_json(str(combined_descriptions_path))
        
        # Step 3: Create splits if requested
        if create_splits:
            self.logger.info("=== Creating Data Splits ===")
            stage_paths = self.config.get_stage_paths('stage3')
            
            # Create splits for codes
            codes_input = str(Path(stage_paths['filtered_path']) / 'specialty_query_dict.json')
            codes_output = str(Path(stage_paths['filtered_path']) / 'specialty_query_splits')
            self.data_splitter.split_data(codes_input, codes_output, self.config.processing.num_chunks)
            
            # Create splits for descriptions
            descriptions_input = str(Path(stage_paths['filtered_path']) / 'specialty_query_code_description_dict.json')
            descriptions_output = str(Path(stage_paths['filtered_path']) / 'specialty_query_code_description_splits')
            self.data_splitter.split_data(descriptions_input, descriptions_output, self.config.processing.num_chunks)
        
        # Step 4: Verify ICD codes
        if mode in ['verify', 'all']:
            self.logger.info("=== Starting ICD Code Verification ===")
            retry_verify = self.verify_icd_codes_stage(
                specialty_query_dict,
                specialty_query_code_description_dict,
                models['gpt-4.1'],
                chunk_index
            )
            results['verify_retry'] = retry_verify
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process ICD generation and verification")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")
    parser.add_argument("--mode", choices=['generate', 'combine', 'verify', 'all'], 
                       default='all', help="Processing mode")
    parser.add_argument("--chunk-index", type=int, help="Chunk index for distributed processing")
    parser.add_argument("--combine-results", action="store_true", default=True, 
                       help="Combine GPT results")
    parser.add_argument("--create-splits", action="store_true", help="Create data splits")
    parser.add_argument("--load-splits", action="store_true", help="Load existing splits")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup configuration and logging
    config = ConfigManager(args.config)
    logger = get_logger("icd_processor", args.log_level, log_dir="logs")
    
    # Create and run processor
    processor = ICDProcessor(config, logger)
    
    try:
        results = processor.run(
            mode=args.mode,
            chunk_index=args.chunk_index,
            combine_results=args.combine_results,
            create_splits=args.create_splits,
            load_splits=args.load_splits
        )
        
        logger.info("ICD processing completed successfully")
        
        # Log results summary
        if 'generate_retry' in results and results['generate_retry']:
            logger.warning(f"Generation retry items: {sum(len(v) for v in results['generate_retry'].values())}")
        
        if 'verify_retry' in results and results['verify_retry']:
            logger.warning(f"Verification retry items: {sum(len(v) for v in results['verify_retry'].values())}")
        
        if 'combined_specialties' in results:
            logger.info(f"Combined data for {results['combined_specialties']} specialties")
        
    except Exception as e:
        logger.error(f"ICD processing failed: {e}")
        raise


if __name__ == '__main__':
    main()