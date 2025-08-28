"""
Configuration management for the medical data pipeline.
"""
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    gpt_4o_version: str = "2024-05-01-preview"
    gpt_4o_deployment: str = "gpt-4o"
    gpt_41_version: str = "2024-12-01-preview"
    gpt_41_deployment: str = "gpt-4.1"
    o4_mini_version: str = "2024-12-01-preview"
    o4_mini_deployment: str = "o4-mini"
    max_tokens: int = 4000
    temperature: float = 0.9
    seed: int = 1337


@dataclass
class PathConfig:
    """Configuration for file paths."""
    base_datasets_path: str = "../../../datasets"
    augmented_datasets_path: str = "../../../datasets/datasets_augmented"
    embeddings_path: str = "../../../datasets/embeddings"
    output_path: str = "../../../datasets/output"
    model_path: str = "../../../../shekhar_tanwar/ICD-ICD-Triplet/model/NovaSearch_stella_en_1.5B_v5/"
    icd_reference_file: str = "../../../../shekhar_tanwar/ICD-ICD-Triplet/dataset/icd10.csv"


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    batch_size: int = 8
    num_chunks: int = 4
    target_queries_per_specialty: int = 250
    paraphrased_threshold: int = 50
    similarity_threshold: float = 0.9
    top_k_positives: int = 10
    num_hard_negatives: int = 50
    retry_attempts: int = 3


@dataclass
class ExecutionConfig:
    """Configuration for execution modes."""
    load_retry: bool = False
    batch_mode: bool = False
    validate_already_classified: bool = False
    load_embeddings_flag: bool = False
    use_gpu: bool = True
    combine_results: bool = False
    create_splits: bool = False
    load_splits: bool = False


class ConfigManager:
    """Manages configuration for the medical data pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.model = ModelConfig()
        self.paths = PathConfig()
        self.processing = ProcessingConfig()
        self.execution = ExecutionConfig()
        
        if os.path.exists(self.config_path):
            self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            for key, value in config_data.get('model', {}).items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
            
            for key, value in config_data.get('paths', {}).items():
                if hasattr(self.paths, key):
                    setattr(self.paths, key, value)
            
            for key, value in config_data.get('processing', {}).items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
            
            for key, value in config_data.get('execution', {}).items():
                if hasattr(self.execution, key):
                    setattr(self.execution, key, value)
                    
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
    
    def save_config(self) -> None:
        """Save current configuration to JSON file."""
        config_data = {
            'model': self.model.__dict__,
            'paths': self.paths.__dict__,
            'processing': self.processing.__dict__,
            'execution': self.execution.__dict__
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
    
    def update_execution_flags(self, **kwargs) -> None:
        """Update execution flags."""
        for key, value in kwargs.items():
            if hasattr(self.execution, key):
                setattr(self.execution, key, value)
    
    def get_stage_paths(self, stage: str) -> Dict[str, str]:
        """Get paths specific to a processing stage."""
        base_path = Path(self.paths.augmented_datasets_path)
        
        stage_paths = {
            'stage2': {
                'input_path': str(base_path / 'datasets_error_analysis/result_evaluation_inhouse_gpt'),
                'output_path': str(base_path / 'augmentation_set4/iteration1'),
                'classification_path': str(base_path / 'augmentation_set3/gpt41_query_clasification_results'),
            },
            'stage3': {
                'input_path': str(base_path / 'final_dataset_v40/splits/splits'),
                'gpt_4o_results': str(base_path / 'icd_sets_v40/gpt_4o_results'),
                'gpt_41_results': str(base_path / 'icd_sets_v40/gpt_41_results'),
                'filtered_path': str(base_path / 'final_dataset_v40/icd_filtered'),
            },
            'stage4': {
                'input_path': str(base_path / 'final_dataset_v40/icd_filtered/filtered_icd_codes'),
                'output_path': str(base_path / 'final_dataset_v40/specialty_verification'),
            },
            'stage5': {
                'input_path': str(base_path / 'final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/filtered_specialty_query_code_results'),
                'output_path': str(base_path / 'triplets_v50_250_queries_10positives_50hn'),
            }
        }
        
        return stage_paths.get(stage, {})


def create_default_config() -> None:
    """Create a default configuration file."""
    config_manager = ConfigManager()
    config_manager.save_config()
    print(f"Default configuration saved to {config_manager.config_path}")


if __name__ == "__main__":
    create_default_config()
