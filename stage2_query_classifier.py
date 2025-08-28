"""
Stage 2 - Step 2: Medical Query Classification
Classifies medical queries as diagnostic, procedural, or exclude.
"""
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from base_processor import BaseProcessor, FileManager, RetryProcessor
from config import ConfigManager
from logging_utils import PipelineLogger, get_logger
from error_handling import handle_errors, validate_inputs, validate_non_empty_string
from azure_utils import AzureModelManager


class ClassificationConfig:
    """Configuration for query classification."""
    
    SYSTEM_PROMPT = """You are a senior medical coding expert specializing in both diagnostic (ICD) and procedural (CPT) classification. Your task is to analyze a user medical query and its associated specialty, extract medically relevant information, and classify the query as either:

    - diagnostic: for queries where the user's intent is to identify or describe a medical condition.
    - procedural: for queries where the user's intent involves a medical procedure or treatment.
    - exclude: for ambiguous, non-medical, or multi-intent queries where confident classification is not possible.
    
    Please follow this chain-of-thought reasoning process:
    
    Step 1: **Preprocessing**
    - Remove names (e.g., "Sara Moore") and address/location fields (e.g., "Santa Monica") from the query.
    - Normalize terms and extract only medically relevant tokens.
    
    Step 2: **Medical Intent Extraction**
    - Identify whether the query expresses:
      - symptoms (e.g., chest pain, headache, back pain)
      - conditions (e.g., hypertension, arthritis)
      - diagnostic tests (e.g., blood test, MRI)
      - procedures or interventions (e.g., surgery, therapy, replacement)
    - Analyze whether the medical target specialty or subspecialty aligns with the query terms.
    
    Step 3: **Code Type Determination**
    - If the query asks about a condition, diagnosis, or symptom: assign **diagnostic**.
    - If the query refers to a procedure, surgery, or therapeutic action: assign **procedural**.
    - If both types of intents are present (e.g., mentions both symptoms and surgery), assign **exclude**
    
    Step 4: **Confidence Check**
    - Only assign "diagnostic" or "procedural" if the query intent is clearly aligned with one category.
    - If unsure or if multiple intents are present, return "exclude".
    
    Final Output Format:
    
    classification: <diagnostic|procedural|exclude>
    
    
    ### Examples
    
    **Example 1:**
    medical_query: "Chest pain"
    target_specialty: "cardiology"
    → classification: diagnostic
    → reason: Mentions a symptom (chest pain) aligned with cardiology; indicates diagnostic evaluation.
    
    **Example 2:**
    medical_query: "ACL reconstruction surgery"
    target_specialty: "orthopedic_surgery"
    → classification: procedural
    → reason: Clearly indicates a surgical procedure aligned with orthopedics.
    
    **Example 3:**
    medical_query: "Dr. Moore for knee replacement"
    target_specialty: "orthopedics"
    → classification: exclude
    → reason: Query mixes a name with procedural content but lacks clarity about user intent or relevant symptoms.
    
    Do not generate ICD or CPT codes. Your only task is query classification. Please only output the label associated with the query, no other text.

    Format Instructions: {format_instructions}  
    medical_query: {medical_query}
    target_specialty: {medical_specialty}
    """


class ClassificationResponse(BaseModel):
    """Pydantic model for classification response."""
    classification: str = Field(description="One of ['diagnostic', 'procedural', 'exclude']")


class DatasetLoader:
    """Loads datasets for query classification."""
    
    def __init__(self, file_manager: FileManager, logger: PipelineLogger):
        self.file_manager = file_manager
        self.logger = logger
    
    def load_batch_datasets(self, batch_path: str) -> Dict[str, List[str]]:
        """Load datasets from batch files."""
        all_files = self.file_manager.list_files(batch_path, '.json')
        specialty_query_dict = {}
        
        for file_path in all_files:
            try:
                data = self.file_manager.load_json(file_path)
                for specialty, queries in data.items():
                    specialty_query_dict[specialty] = queries
            except Exception as e:
                self.logger.warning(f"Failed to load batch file {file_path}: {e}")
        
        self.logger.info(f'Loaded {len(specialty_query_dict)} specialties from batch files')
        return specialty_query_dict
    
    def load_single_dataset(self, file_path: str) -> Dict[str, List[str]]:
        """Load dataset from single file."""
        data = self.file_manager.load_json(file_path)
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        
        self.logger.info(f'Loaded {len(data)} specialties from single file')
        return data
    
    def load_datasets(self, batch_mode: bool, batch_path: Optional[str], single_file_path: str) -> Dict[str, List[str]]:
        """
        Load datasets based on mode.
        
        Args:
            batch_mode: Whether to load from batch files
            batch_path: Path to batch files directory
            single_file_path: Path to single file
        
        Returns:
            Dictionary of specialty -> queries mapping
        """
        if batch_mode:
            if not batch_path:
                raise ValueError("batch_path required when batch_mode=True")
            self.logger.info('Loading Synthetic Queries for Specialty Classification')
            return self.load_batch_datasets(batch_path)
        else:
            self.logger.info('Loading UES Queries for Specialty Classification')
            return self.load_single_dataset(single_file_path)


class QueryClassifier:
    """Classifies medical queries using Azure OpenAI."""
    
    def __init__(self, azure_manager: AzureModelManager, logger: PipelineLogger):
        self.azure_manager = azure_manager
        self.logger = logger
        self.output_parser = PydanticOutputParser(pydantic_object=ClassificationResponse)
        self.prompt_template = PromptTemplate.from_template(ClassificationConfig.SYSTEM_PROMPT)
    
    @handle_errors(continue_on_error=True)
    def classify_query(self, model, medical_specialty: str, medical_query: str) -> Optional[str]:
        """
        Classify a medical query.
        
        Args:
            model: Azure OpenAI model instance
            medical_specialty: Medical specialty context
            medical_query: Query to classify
        
        Returns:
            Classification result or None if failed
        """
        chain = self.prompt_template | model | self.output_parser
        
        result = chain.invoke(input={
            "medical_specialty": medical_specialty,
            "medical_query": medical_query,
            "format_instructions": self.output_parser.get_format_instructions()
        })
        
        return result.classification if result else None


class QueryClassificationProcessor(BaseProcessor):
    """Main processor for query classification."""
    
    def __init__(self, config: ConfigManager, logger: PipelineLogger):
        super().__init__(config, logger, "STAGE2_QUERY_CLASSIFICATION")
        self.file_manager = FileManager(logger)
        self.dataset_loader = DatasetLoader(self.file_manager, logger)
        self.query_classifier = QueryClassifier(self.azure_manager, logger)
        self.retry_processor = RetryProcessor(logger, config.processing.retry_attempts)
    
    def filter_diagnostic_queries(self, 
                                specialty_query_dict: Dict[str, List[str]], 
                                model,
                                output_path: str,
                                validate_processed: bool = False,
                                processed_specialties_path: str = "") -> Dict[str, float]:
        """
        Filter queries to keep only diagnostic ones.
        
        Args:
            specialty_query_dict: Dictionary of specialty -> queries
            model: Azure OpenAI model for classification
            output_path: Path to save filtered results
            validate_processed: Whether to skip already processed specialties
            processed_specialties_path: Path to check for processed specialties
        
        Returns:
            Dictionary of specialty -> filtering metrics
        """
        from tqdm import tqdm
        
        specialty_query_metrics = {}
        processed_specialties = []
        
        # Get already processed specialties
        if validate_processed and processed_specialties_path:
            processed_specialties = self.file_manager.get_processed_files(processed_specialties_path)
        
        self.file_manager.ensure_directory(output_path)
        
        for specialty, queries in tqdm(specialty_query_dict.items(), desc="Classifying queries"):
            if validate_processed and specialty in processed_specialties:
                self.logger.debug(f"Skipping already processed specialty: {specialty}")
                continue
            
            self.logger.info(f'Processing Specialty: {specialty}')
            
            filtered_queries = []
            for query in tqdm(queries, desc=f"Processing {specialty}", leave=False):
                classification = self.retry_processor.execute_with_retry(
                    self.query_classifier.classify_query,
                    f"{specialty}_{query[:50]}",  # Truncate for logging
                    model, specialty, query
                )
                
                if classification == 'diagnostic':
                    filtered_queries.append(query)
            
            # Calculate filtering metrics
            original_count = len(queries)
            filtered_count = len(filtered_queries)
            filter_percentage = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
            
            specialty_query_metrics[specialty] = filter_percentage
            
            # Save filtered results
            specialty_query_class = {specialty: filtered_queries}
            output_file = Path(output_path) / f'{specialty}.json'
            self.file_manager.save_json(specialty_query_class, str(output_file))
            
            self.logger.info(f'Processed Specialty: {specialty} - '
                           f'Kept {filtered_count}/{original_count} queries '
                           f'({100-filter_percentage:.1f}% retained)')
            
            time.sleep(0.01)  # Brief pause
        
        return specialty_query_metrics
    
    @validate_inputs(
        model_name=lambda x: x in ['gpt-4o', 'gpt-4.1']
    )
    def process(self,
                batch_mode: bool = False,
                batch_path: Optional[str] = None,
                single_file_path: str = "",
                model_name: str = "gpt-4.1",
                validate_processed: bool = False,
                processed_specialties_path: str = "",
                output_path: str = "") -> Dict[str, float]:
        """
        Process query classification.
        
        Args:
            batch_mode: Whether to load from batch files
            batch_path: Path to batch files directory
            single_file_path: Path to single dataset file
            model_name: Name of model to use
            validate_processed: Whether to skip processed specialties
            processed_specialties_path: Path to check processed specialties
            output_path: Path to save results
        
        Returns:
            Dictionary of filtering metrics by specialty
        """
        # Initialize model
        model = self.azure_manager.initialize_model(model_name)
        
        # Load datasets
        specialty_query_dict = self.dataset_loader.load_datasets(
            batch_mode, batch_path, single_file_path
        )
        
        # Set output path from config if not provided
        if not output_path:
            stage_paths = self.config.get_stage_paths('stage2')
            output_path = stage_paths['classification_path']
        
        # Filter queries
        self.logger.info(f'Filtering Queries By Specialty using {model_name}')
        metrics = self.filter_diagnostic_queries(
            specialty_query_dict,
            model,
            output_path,
            validate_processed,
            processed_specialties_path
        )
        
        # Save metrics
        metrics_file = Path(output_path).parent / f'classification_metrics_{model_name}.json'
        self.file_manager.save_json(metrics, str(metrics_file))
        self.logger.info(f'Saved classification metrics to {metrics_file}')
        
        return metrics


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify medical queries")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")
    parser.add_argument("--batch-mode", action="store_true", help="Load from batch files")
    parser.add_argument("--batch-path", type=str, help="Path to batch files directory")
    parser.add_argument("--single-file", type=str, help="Path to single dataset file")
    parser.add_argument("--model-name", choices=['gpt-4o', 'gpt-4.1'], default='gpt-4.1', help="Model to use")
    parser.add_argument("--validate-processed", action="store_true", help="Skip processed specialties")
    parser.add_argument("--processed-path", type=str, default="", help="Path to processed specialties")
    parser.add_argument("--output-path", type=str, default="", help="Output path for results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.batch_mode and not args.single_file:
        parser.error("Either --batch-mode with --batch-path or --single-file is required")
    
    if args.batch_mode and not args.batch_path:
        parser.error("--batch-path is required when using --batch-mode")
    
    # Setup configuration and logging
    config = ConfigManager(args.config)
    logger = get_logger("query_classifier", args.log_level, log_dir="logs")
    
    # Create and run processor
    processor = QueryClassificationProcessor(config, logger)
    
    try:
        metrics = processor.run(
            batch_mode=args.batch_mode,
            batch_path=args.batch_path,
            single_file_path=args.single_file or "",
            model_name=args.model_name,
            validate_processed=args.validate_processed,
            processed_specialties_path=args.processed_path,
            output_path=args.output_path
        )
        
        logger.info("Query classification completed successfully")
        
        # Log summary statistics
        total_specialties = len(metrics)
        avg_filter_rate = sum(metrics.values()) / total_specialties if total_specialties > 0 else 0
        logger.info(f"Processed {total_specialties} specialties")
        logger.info(f"Average filter rate: {avg_filter_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        raise


if __name__ == '__main__':
    main()
