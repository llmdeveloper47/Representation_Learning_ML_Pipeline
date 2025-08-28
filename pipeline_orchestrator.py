"""
Medical Data Pipeline Orchestrator
Coordinates execution of all pipeline stages with proper configuration management.
"""
import os
import sys
import argparse
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from base_processor import ProcessingPipeline
from config import ConfigManager
from logging_utils import PipelineLogger, get_logger, setup_default_logging
from error_handling import ErrorCollector, graceful_shutdown

# Import stage processors
from stage2_query_generator import QueryGenerationProcessor
from stage2_query_classifier import QueryClassificationProcessor  
from stage3_icd_processor import ICDProcessor
from stage4_specialty_verifier import SpecialtyVerificationProcessor
from hyperparameter_tuner import HyperparameterTuningProcessor


class PipelineOrchestrator:
    """Orchestrates the execution of the complete medical data pipeline."""
    
    def __init__(self, config_path: str = "config.json", log_level: str = "INFO"):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        self.config = ConfigManager(config_path)
        self.logger = get_logger("pipeline_orchestrator", log_level, log_dir="logs")
        self.error_collector = ErrorCollector(self.logger)
        
        # Setup graceful shutdown
        graceful_shutdown(self.logger, self.error_collector)
        
        # Initialize pipeline
        self.pipeline = ProcessingPipeline(self.config, self.logger)
        self.processors = {}
        
        self._initialize_processors()
    
    def _initialize_processors(self) -> None:
        """Initialize all stage processors."""
        try:
            self.processors = {
                'query_generation': QueryGenerationProcessor(self.config, self.logger),
                'query_classification': QueryClassificationProcessor(self.config, self.logger),
                'icd_processing': ICDProcessor(self.config, self.logger),
                'specialty_verification': SpecialtyVerificationProcessor(self.config, self.logger),
                'hyperparameter_tuning': HyperparameterTuningProcessor(self.config, self.logger)
            }
            self.logger.info("All processors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def run_stage2_query_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Run Stage 2: Query Generation and Classification.
        
        Args:
            **kwargs: Stage-specific configuration parameters
        
        Returns:
            Results from Stage 2 processing
        """
        self.logger.log_processing_start("STAGE2", "Query Generation and Classification")
        
        results = {}
        
        # Stage 2.1: Query Generation
        if kwargs.get('run_generation', True):
            self.logger.info("Running query generation...")
            generation_params = {
                'load_retry': kwargs.get('load_retry', False),
                'dataset_type': kwargs.get('dataset_type', 'pickle'),
                'model_name': kwargs.get('generation_model', 'gpt-4o'),
                'chunk_index': kwargs.get('chunk_index')
            }
            
            generation_results = self.processors['query_generation'].run(**generation_params)
            results['generation'] = generation_results
        
        # Stage 2.2: Query Classification
        if kwargs.get('run_classification', True):
            self.logger.info("Running query classification...")
            classification_params = {
                'batch_mode': kwargs.get('batch_mode', False),
                'batch_path': kwargs.get('batch_path'),
                'single_file_path': kwargs.get('single_file_path', ''),
                'model_name': kwargs.get('classification_model', 'gpt-4.1'),
                'validate_processed': kwargs.get('validate_processed', False),
                'processed_specialties_path': kwargs.get('processed_specialties_path', ''),
                'output_path': kwargs.get('classification_output_path', '')
            }
            
            classification_results = self.processors['query_classification'].run(**classification_params)
            results['classification'] = classification_results
        
        self.logger.log_processing_end("STAGE2", "Query Generation and Classification")
        return results
    
    def run_stage3_icd_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Run Stage 3: ICD Code Generation and Verification.
        
        Args:
            **kwargs: Stage-specific configuration parameters
        
        Returns:
            Results from Stage 3 processing
        """
        self.logger.log_processing_start("STAGE3", "ICD Code Generation and Verification")
        
        icd_params = {
            'mode': kwargs.get('mode', 'all'),
            'chunk_index': kwargs.get('chunk_index'),
            'combine_results': kwargs.get('combine_results', True),
            'create_splits': kwargs.get('create_splits', False),
            'load_splits': kwargs.get('load_splits', False)
        }
        
        results = self.processors['icd_processing'].run(**icd_params)
        
        self.logger.log_processing_end("STAGE3", "ICD Code Generation and Verification")
        return results
    
    def run_stage4_specialty_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Run Stage 4: Specialty Verification.
        
        Args:
            **kwargs: Stage-specific configuration parameters
        
        Returns:
            Results from Stage 4 processing
        """
        self.logger.log_processing_start("STAGE4", "Specialty Verification")
        
        specialty_params = {
            'mode': kwargs.get('mode', 'all'),
            'chunk_index': kwargs.get('chunk_index'),
            'combine_datasets': kwargs.get('combine_datasets', True),
            'create_splits': kwargs.get('create_splits', False),
            'model_name': kwargs.get('model_name', 'gpt-4.1')
        }
        
        results = self.processors['specialty_verification'].run(**specialty_params)
        
        self.logger.log_processing_end("STAGE4", "Specialty Verification")
        return results
    
    def run_hyperparameter_tuning(self, **kwargs) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            **kwargs: Hyperparameter tuning configuration parameters
        
        Returns:
            Results from hyperparameter tuning
        """
        self.logger.log_processing_start("HYPERPARAMETER_TUNING", "Model Hyperparameter Optimization")
        
        tuning_params = {
            'train_path': kwargs.get('train_path', ''),
            'eval_path': kwargs.get('eval_path', ''),
            'n_trials': kwargs.get('n_trials', 10),
            'timeout': kwargs.get('timeout'),
            'sample_fraction': kwargs.get('sample_fraction'),
            'output_dir': kwargs.get('output_dir', 'hyperparameter_results'),
            'study_name': kwargs.get('study_name'),
            'create_samples': kwargs.get('create_samples', True)
        }
        
        results = self.processors['hyperparameter_tuning'].run(**tuning_params)
        
        self.logger.log_processing_end("HYPERPARAMETER_TUNING", "Model Hyperparameter Optimization")
        return results
    
    def run_full_pipeline(self, stages: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run the complete pipeline or specified stages.
        
        Args:
            stages: List of stages to run ('stage2', 'stage3', 'stage4', 'hyperparameter')
            **kwargs: Configuration parameters for all stages
        
        Returns:
            Complete pipeline results
        """
        if stages is None:
            stages = ['stage2', 'stage3', 'stage4']
        
        self.logger.log_processing_start("FULL_PIPELINE", f"Running stages: {', '.join(stages)}")
        
        all_results = {}
        
        try:
            # Stage 2: Query processing
            if 'stage2' in stages:
                stage2_results = self.run_stage2_query_pipeline(**kwargs.get('stage2', {}))
                all_results['stage2'] = stage2_results
            
            # Stage 3: ICD processing
            if 'stage3' in stages:
                stage3_results = self.run_stage3_icd_pipeline(**kwargs.get('stage3', {}))
                all_results['stage3'] = stage3_results
            
            # Stage 4: Specialty verification
            if 'stage4' in stages:
                stage4_results = self.run_stage4_specialty_pipeline(**kwargs.get('stage4', {}))
                all_results['stage4'] = stage4_results
            
            # Hyperparameter tuning
            if 'hyperparameter' in stages:
                hp_results = self.run_hyperparameter_tuning(**kwargs.get('hyperparameter', {}))
                all_results['hyperparameter_tuning'] = hp_results
            
            # Save overall pipeline results
            self._save_pipeline_results(all_results)
            
            self.logger.log_processing_end("FULL_PIPELINE", f"Completed stages: {', '.join(stages)}")
            
        except Exception as e:
            self.error_collector.add_error("FULL_PIPELINE", e)
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return all_results
    
    def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"pipeline_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_json_serializable(results)
        
        try:
            import json
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Pipeline results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline results: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        error_summary = self.error_collector.get_error_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config.config_path,
            'processors': list(self.processors.keys()),
            'error_summary': error_summary,
            'total_errors': error_summary['total_errors'],
            'total_warnings': error_summary['total_warnings']
        }
    
    def validate_configuration(self) -> bool:
        """Validate pipeline configuration."""
        try:
            # Check required paths exist
            required_paths = [
                self.config.paths.base_datasets_path,
                self.config.paths.augmented_datasets_path
            ]
            
            for path in required_paths:
                if not Path(path).exists():
                    self.logger.error(f"Required path does not exist: {path}")
                    return False
            
            # Check ICD reference file
            if not Path(self.config.paths.icd_reference_file).exists():
                self.logger.error(f"ICD reference file not found: {self.config.paths.icd_reference_file}")
                return False
            
            # Check model path
            if not Path(self.config.paths.model_path).exists():
                self.logger.warning(f"Model path not found: {self.config.paths.model_path}")
            
            self.logger.info("Configuration validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


def create_default_config_file() -> None:
    """Create a default configuration file."""
    from config import create_default_config
    create_default_config()
    print("Default configuration file created: config.json")


def main():
    """Main execution function with comprehensive CLI interface."""
    parser = argparse.ArgumentParser(
        description="Medical Data Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline_orchestrator.py --stages stage2 stage3 stage4

  # Run specific stage with custom parameters
  python pipeline_orchestrator.py --stages stage2 --stage2-generation-model gpt-4o

  # Run with chunk processing
  python pipeline_orchestrator.py --stages stage3 --chunk-index 0

  # Create default configuration
  python pipeline_orchestrator.py --create-config

  # Validate configuration only
  python pipeline_orchestrator.py --validate-only
        """
    )
    
    # General arguments
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default="INFO", help="Logging level")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    
    # Pipeline control
    parser.add_argument("--stages", nargs='+', 
                       choices=['stage2', 'stage3', 'stage4', 'hyperparameter'],
                       help="Stages to run")
    parser.add_argument("--chunk-index", type=int, help="Chunk index for distributed processing")
    
    # Stage 2 arguments
    stage2_group = parser.add_argument_group('Stage 2 - Query Processing')
    stage2_group.add_argument("--stage2-generation-model", choices=['gpt-4o', 'gpt-4.1'], 
                             default='gpt-4o', help="Model for query generation")
    stage2_group.add_argument("--stage2-classification-model", choices=['gpt-4o', 'gpt-4.1'], 
                             default='gpt-4.1', help="Model for query classification")
    stage2_group.add_argument("--stage2-load-retry", action="store_true", 
                             help="Load specialties from retry list")
    stage2_group.add_argument("--stage2-batch-mode", action="store_true", 
                             help="Use batch mode for classification")
    stage2_group.add_argument("--stage2-batch-path", type=str, 
                             help="Path to batch files for classification")
    stage2_group.add_argument("--stage2-single-file", type=str, 
                             help="Single file path for classification")
    
    # Stage 3 arguments
    stage3_group = parser.add_argument_group('Stage 3 - ICD Processing')
    stage3_group.add_argument("--stage3-mode", choices=['generate', 'combine', 'verify', 'all'], 
                             default='all', help="ICD processing mode")
    stage3_group.add_argument("--stage3-combine-results", action="store_true", default=True,
                             help="Combine GPT results")
    stage3_group.add_argument("--stage3-create-splits", action="store_true", 
                             help="Create data splits")
    
    # Stage 4 arguments
    stage4_group = parser.add_argument_group('Stage 4 - Specialty Verification')
    stage4_group.add_argument("--stage4-mode", choices=['combine', 'verify', 'all'], 
                             default='all', help="Specialty verification mode")
    stage4_group.add_argument("--stage4-model", choices=['gpt-4o', 'gpt-4.1'], 
                             default='gpt-4.1', help="Model for specialty verification")
    stage4_group.add_argument("--stage4-combine-datasets", action="store_true", default=True,
                             help="Combine filtered datasets")
    stage4_group.add_argument("--stage4-create-splits", action="store_true", 
                             help="Create data splits")
    
    # Hyperparameter tuning arguments
    hp_group = parser.add_argument_group('Hyperparameter Tuning')
    hp_group.add_argument("--hp-train-path", type=str, help="Training dataset path")
    hp_group.add_argument("--hp-eval-path", type=str, help="Evaluation dataset path")
    hp_group.add_argument("--hp-n-trials", type=int, default=10, help="Number of optimization trials")
    hp_group.add_argument("--hp-timeout", type=int, help="Timeout in seconds")
    hp_group.add_argument("--hp-sample-fraction", type=float, help="Data sampling fraction")
    hp_group.add_argument("--hp-output-dir", type=str, default="hyperparameter_results", 
                         help="Output directory")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_default_config_file()
        return
    
    # Setup logging
    setup_default_logging()
    
    # Initialize orchestrator
    try:
        orchestrator = PipelineOrchestrator(args.config, args.log_level)
    except Exception as e:
        print(f"Failed to initialize pipeline orchestrator: {e}")
        sys.exit(1)
    
    # Validate configuration
    if not orchestrator.validate_configuration():
        orchestrator.logger.error("Configuration validation failed")
        sys.exit(1)
    
    if args.validate_only:
        orchestrator.logger.info("Configuration validation completed successfully")
        return
    
    # Prepare stage configurations
    stage_configs = {
        'stage2': {
            'run_generation': True,
            'run_classification': True,
            'generation_model': args.stage2_generation_model,
            'classification_model': args.stage2_classification_model,
            'load_retry': args.stage2_load_retry,
            'batch_mode': args.stage2_batch_mode,
            'batch_path': args.stage2_batch_path,
            'single_file_path': args.stage2_single_file or '',
            'chunk_index': args.chunk_index
        },
        'stage3': {
            'mode': args.stage3_mode,
            'chunk_index': args.chunk_index,
            'combine_results': args.stage3_combine_results,
            'create_splits': args.stage3_create_splits
        },
        'stage4': {
            'mode': args.stage4_mode,
            'chunk_index': args.chunk_index,
            'model_name': args.stage4_model,
            'combine_datasets': args.stage4_combine_datasets,
            'create_splits': args.stage4_create_splits
        },
        'hyperparameter': {
            'train_path': args.hp_train_path or '',
            'eval_path': args.hp_eval_path or '',
            'n_trials': args.hp_n_trials,
            'timeout': args.hp_timeout,
            'sample_fraction': args.hp_sample_fraction,
            'output_dir': args.hp_output_dir
        }
    }
    
    # Run pipeline
    try:
        if args.stages:
            results = orchestrator.run_full_pipeline(args.stages, **stage_configs)
        else:
            orchestrator.logger.error("No stages specified. Use --stages to specify which stages to run.")
            parser.print_help()
            sys.exit(1)
        
        # Print final status
        status = orchestrator.get_pipeline_status()
        orchestrator.logger.info("Pipeline execution completed!")
        orchestrator.logger.info(f"Total errors: {status['total_errors']}")
        orchestrator.logger.info(f"Total warnings: {status['total_warnings']}")
        
        if status['total_errors'] > 0:
            orchestrator.logger.warning("Pipeline completed with errors. Check logs for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        orchestrator.logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        orchestrator.logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()