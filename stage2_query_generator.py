"""
Stage 2 - Step 1: Medical Query Generation
Generates medical queries for specialties using GPT models.
"""
import os
import json
import time
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from pydantic import BaseModel, Field

from base_processor import BaseProcessor, FileManager, RetryProcessor
from config import ConfigManager
from logging_utils import PipelineLogger, get_logger
from error_handling import handle_errors, validate_inputs, validate_file_exists, validate_positive_integer
from azure_utils import AzureModelManager


class QueryGenerationConfig:
    """Configuration for query generation."""
    
    SYSTEM_PROMPT = """
    You are a clinical data generation expert specializing in constructing medically valid queries for natural language medical search engines.
    
    Your task is to generate realistic queries that a patient might enter into a healthcare application or search engine. The queries must reflect **symptoms, conditions, procedures and treatments, or diagnostic concerns** related to the target medical specialty and subspecialty.
    
    Instructions:
    1. avoid procedural and treatment queries that would result in very broad range of possible diagnosis codes when billed in a claim
    2. avoid procedural and treatment queries that are related to a very broad range of condition and diseases that may have very broad range of possible diagnosis codes
    3. Do NOT include any procedural language (e.g., "therapy", "surgery", "MRI", "replacement", "treatment", "rehab").
    4. Focus on real-world concerns a patient might ask when unsure of their condition 
    5. Include both short-form and long-form natural language queries, i.e 2 to 8 words long queries.
    6. Do not include doctor names or clinic addresses.
    7. Use diverse phrasing and vocabulary across examples.
    

    Task:
    Generate 200 queries related to the above specialty.

    Only output the query, no other text.

    Format Instructions:
    {format_instructions}
                        
    medical_specialty: {medical_specialty}
    """


class MedicalQuery(BaseModel):
    """Pydantic model for medical queries."""
    queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")


class SpecialtyDataLoader:
    """Loads specialty data for query generation."""
    
    def __init__(self, file_manager: FileManager, logger: PipelineLogger):
        self.file_manager = file_manager
        self.logger = logger
    
    def load_specialty_datasets(self, config: ConfigManager) -> List[str]:
        """Load specialties from dataset files."""
        stage_paths = config.get_stage_paths('stage2')
        input_path = stage_paths['input_path']
        
        # Define dataset file paths
        dataset_files = [
            f'{input_path}/gpt_specialty_query_dict_1.json',
            f'{input_path}/gpt_specialty_query_dict_2.json', 
            f'{input_path}/gpt_specialty_query_dict_3.json',
            f'{input_path}/gpt_specialty_query_dict_4.json'
        ]
        
        combined_data = {}
        
        for file_path in dataset_files:
            try:
                data = self.file_manager.load_json(file_path)
                for specialty, queries in data.items():
                    if specialty in combined_data:
                        combined_data[specialty].update(queries)
                    else:
                        combined_data[specialty] = queries.copy()
            except Exception as e:
                self.logger.warning(f"Could not load dataset file {file_path}: {e}")
        
        all_specialties = list(combined_data.keys())
        self.logger.info(f'Total Specialties Covered: {len(all_specialties)}')
        
        return all_specialties


class QueryGenerator:
    """Generates medical queries using Azure OpenAI."""
    
    def __init__(self, azure_manager: AzureModelManager, logger: PipelineLogger):
        self.azure_manager = azure_manager
        self.logger = logger
        self.output_parser = CommaSeparatedListOutputParser()
        self.prompt_template = PromptTemplate.from_template(QueryGenerationConfig.SYSTEM_PROMPT)
    
    @handle_errors(continue_on_error=True)
    def generate_queries(self, model, medical_specialty: str) -> List[str]:
        """
        Generate queries for a medical specialty.
        
        Args:
            model: Azure OpenAI model instance
            medical_specialty: Medical specialty to generate queries for
        
        Returns:
            List of generated queries
        """
        chain = self.prompt_template | model | self.output_parser
        
        result = chain.invoke(input={
            "medical_specialty": medical_specialty,
            "format_instructions": self.output_parser.get_format_instructions()
        })
        
        self.logger.debug(f"Generated {len(result)} queries for {medical_specialty}")
        return result


class QueryGenerationProcessor(BaseProcessor):
    """Main processor for medical query generation."""
    
    def __init__(self, config: ConfigManager, logger: PipelineLogger):
        super().__init__(config, logger, "STAGE2_QUERY_GENERATION")
        self.file_manager = FileManager(logger)
        self.data_loader = SpecialtyDataLoader(self.file_manager, logger)
        self.query_generator = QueryGenerator(self.azure_manager, logger)
        self.retry_processor = RetryProcessor(logger, config.processing.retry_attempts)
    
    def load_retry_list(self, dataset_type: str = 'pickle') -> List[str]:
        """Load list of specialties to retry."""
        if dataset_type == 'csv':
            import pandas as pd
            dataset_path = './final_stats_less_200_diagnostic_quries.csv'
            df = pd.read_csv(dataset_path)
            return list(df.iloc[:, 1:]['Specialties'])
        else:
            retry_path = self.config.get_stage_paths('stage2')['output_path'] + '/../retry_list_v1.pkl'
            try:
                return self.file_manager.load_pickle(retry_path)
            except Exception as e:
                self.logger.warning(f"Could not load retry list: {e}")
                return []
    
    @validate_inputs(
        model_name=lambda x: x in ['gpt-4o', 'gpt-4.1'],
        chunk_index=validate_positive_integer
    )
    def process(self, 
                load_retry: bool = False,
                dataset_type: str = 'pickle', 
                model_name: str = 'gpt-4o',
                chunk_index: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Process query generation for specialties.
        
        Args:
            load_retry: Whether to load specialties from retry list
            dataset_type: Type of dataset to load ('csv' or 'pickle')
            model_name: Name of model to use for generation
            chunk_index: Optional chunk index for distributed processing
        
        Returns:
            Dictionary of retry items
        """
        # Initialize model
        model = self.azure_manager.initialize_model(model_name)
        
        # Load specialties
        if load_retry:
            all_specialties = self.load_retry_list(dataset_type)
            self.logger.info(f'Loading {len(all_specialties)} specialties from retry list')
        else:
            all_specialties = self.data_loader.load_specialty_datasets(self.config)
            self.logger.info(f'Loading {len(all_specialties)} specialties from datasets')
        
        # Filter out already processed specialties
        stage_paths = self.config.get_stage_paths('stage2')
        output_path = stage_paths['output_path']
        self.file_manager.ensure_directory(output_path)
        
        processed_specialties = self.file_manager.get_processed_files(output_path)
        remaining_specialties = [s for s in all_specialties if s not in processed_specialties]
        
        self.logger.info(f'Processing {len(remaining_specialties)} remaining specialties')
        
        # Process specialties
        from tqdm import tqdm
        
        for i, medical_specialty in enumerate(tqdm(remaining_specialties, desc="Generating queries")):
            self.logger.debug(f'Processing Specialty: {medical_specialty}')
            
            try:
                # Generate queries with multiple attempts for diversity
                queries_1 = self.retry_processor.execute_with_retry(
                    self.query_generator.generate_queries,
                    f"{medical_specialty}_1",
                    model, medical_specialty
                )
                
                queries_2 = self.retry_processor.execute_with_retry(
                    self.query_generator.generate_queries,
                    f"{medical_specialty}_2", 
                    model, medical_specialty
                )
                
                if queries_1 and queries_2:
                    # Combine and deduplicate queries
                    final_query_set = list(set(queries_1 + queries_2))
                    
                    self.logger.info(f'Processed Specialty: {medical_specialty} - '
                                   f'Generated {len(final_query_set)} unique queries')
                    
                    # Save results
                    specialty_query_dict = {medical_specialty: final_query_set}
                    output_file = Path(output_path) / f'{medical_specialty}.json'
                    self.file_manager.save_json(specialty_query_dict, str(output_file))
                
                else:
                    self.logger.error(f'Failed to generate queries for: {medical_specialty}')
                
                # Brief pause to avoid rate limiting
                time.sleep(0.01)
                
                # Log progress periodically
                if (i + 1) % 10 == 0:
                    self.resource_monitor.log_resource_usage(f"specialty_{i+1}")
                
            except Exception as e:
                self.logger.error(f'Error processing specialty {medical_specialty}: {e}')
                self.error_collector.add_error(self.stage_name, e, {'specialty': medical_specialty})
        
        # Save retry list
        retry_items = self.retry_processor.get_retry_items()
        if retry_items:
            retry_path = Path(output_path).parent / 'retry_list_v1.pkl'
            self.file_manager.save_pickle(list(retry_items.values())[0], str(retry_path))
            self.logger.info(f'Saved {len(retry_items)} retry items')
        
        return retry_items


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate medical queries for specialties")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")
    parser.add_argument("--load-retry", action="store_true", help="Load specialties from retry list")
    parser.add_argument("--dataset-type", choices=['csv', 'pickle'], default='pickle', help="Dataset type for retry list")
    parser.add_argument("--model-name", choices=['gpt-4o', 'gpt-4.1'], default='gpt-4o', help="Model to use")
    parser.add_argument("--chunk-index", type=int, help="Chunk index for distributed processing")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup configuration and logging
    config = ConfigManager(args.config)
    logger = get_logger("query_generator", args.log_level, log_dir="logs")
    
    # Create and run processor
    processor = QueryGenerationProcessor(config, logger)
    
    try:
        retry_items = processor.run(
            load_retry=args.load_retry,
            dataset_type=args.dataset_type,
            model_name=args.model_name,
            chunk_index=args.chunk_index
        )
        
        logger.info("Query generation completed successfully")
        if retry_items:
            logger.warning(f"Some specialties failed and may need retry: {len(retry_items)}")
        
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        raise


if __name__ == '__main__':
    main()