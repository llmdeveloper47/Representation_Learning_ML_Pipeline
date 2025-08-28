"""
Base classes for pipeline processing components.
"""
import os
import json
import time
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from collections import defaultdict

from config import ConfigManager
from logging_utils import PipelineLogger
from error_handling import ErrorCollector, handle_errors, safe_file_operation, ResourceMonitor
from azure_utils import AzureModelManager


class BaseProcessor(ABC):
    """Base class for all pipeline processors."""
    
    def __init__(self, config: ConfigManager, logger: PipelineLogger, stage_name: str):
        self.config = config
        self.logger = logger
        self.stage_name = stage_name
        self.error_collector = ErrorCollector(logger)
        self.resource_monitor = ResourceMonitor(logger)
        self.azure_manager = AzureModelManager(config, logger)
        
    def setup(self) -> None:
        """Setup the processor (called before processing)."""
        self.logger.log_processing_start(self.stage_name, self.__class__.__name__)
        self.resource_monitor.start_monitoring()
        
    def cleanup(self) -> None:
        """Cleanup after processing."""
        self.resource_monitor.log_resource_usage(f"{self.stage_name}_end")
        
        if self.error_collector.has_errors():
            error_summary = self.error_collector.get_error_summary()
            self.logger.warning(f"Processing completed with {error_summary['total_errors']} errors")
        
        self.logger.log_processing_end(self.stage_name, self.__class__.__name__)
    
    @abstractmethod
    def process(self, **kwargs) -> Any:
        """Main processing method to be implemented by subclasses."""
        pass
    
    def run(self, **kwargs) -> Any:
        """Run the processor with setup and cleanup."""
        try:
            self.setup()
            result = self.process(**kwargs)
            return result
        finally:
            self.cleanup()


class FileManager:
    """Manages file operations for the pipeline."""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
    
    @safe_file_operation("load JSON")
    def load_json(self, filepath: str) -> Dict:
        """Load data from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.logger.log_file_operation("load", filepath, True)
        return data
    
    @safe_file_operation("save JSON")
    def save_json(self, data: Dict, filepath: str, indent: int = 4) -> None:
        """Save data to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        self.logger.log_file_operation("save", filepath, True)
    
    @safe_file_operation("load pickle")
    def load_pickle(self, filepath: str) -> Any:
        """Load data from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.logger.log_file_operation("load", filepath, True)
        return data
    
    @safe_file_operation("save pickle")
    def save_pickle(self, data: Any, filepath: str) -> None:
        """Save data to pickle file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        self.logger.log_file_operation("save", filepath, True)
    
    def list_files(self, directory: str, extension: str = ".json") -> List[str]:
        """List files in directory with given extension."""
        dir_path = Path(directory)
        if not dir_path.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return []
        
        files = [str(f) for f in dir_path.glob(f"*{extension}")]
        self.logger.info(f"Found {len(files)} {extension} files in {directory}")
        return files
    
    def ensure_directory(self, directory: str) -> None:
        """Ensure directory exists."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Ensured directory exists: {directory}")
    
    def get_processed_files(self, directory: str, extension: str = ".json") -> List[str]:
        """Get list of already processed files (without extension)."""
        files = self.list_files(directory, extension)
        return [Path(f).stem for f in files]


class DataSplitter:
    """Handles data splitting operations."""
    
    def __init__(self, logger: PipelineLogger, file_manager: FileManager):
        self.logger = logger
        self.file_manager = file_manager
    
    def split_data(self, input_file: str, output_dir: str, num_chunks: int, 
                   filename_prefix: str = "split") -> List[Dict]:
        """
        Split input data into chunks.
        
        Args:
            input_file: Path to input file
            output_dir: Directory for output chunks
            num_chunks: Number of chunks to create
            filename_prefix: Prefix for output filenames
        
        Returns:
            List of chunk data
        """
        # Check if splits already exist
        splits_dir = Path(output_dir) / "splits"
        existing_files = self.file_manager.list_files(str(splits_dir), ".json")
        
        if len(existing_files) == num_chunks:
            self.logger.info(f"Loading existing {num_chunks} splits from {splits_dir}")
            chunks = []
            for file_path in existing_files:
                chunks.append(self.file_manager.load_json(file_path))
            return chunks
        
        # Create new splits
        self.logger.info(f"Creating {num_chunks} splits from {input_file}")
        self.file_manager.ensure_directory(str(splits_dir))
        
        # Load input data
        data = self.file_manager.load_json(input_file)
        
        # Calculate chunk size
        if isinstance(data, dict):
            items = list(data.keys())
            chunk_size = len(items) // num_chunks + (1 if len(items) % num_chunks else 0)
            
            chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, len(items))
                
                chunk_items = items[start_idx:end_idx]
                chunk_data = {item: data[item] for item in chunk_items}
                
                # Save chunk
                filename = Path(input_file).stem
                chunk_file = splits_dir / f"{filename}_{filename_prefix}_{i}.json"
                self.file_manager.save_json(chunk_data, str(chunk_file))
                
                chunks.append(chunk_data)
                self.logger.debug(f"Created chunk {i}: {len(chunk_items)} items")
            
        elif isinstance(data, list):
            chunk_size = len(data) // num_chunks + (1 if len(data) % num_chunks else 0)
            
            chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, len(data))
                
                chunk_data = data[start_idx:end_idx]
                
                # Save chunk
                filename = Path(input_file).stem
                chunk_file = splits_dir / f"{filename}_{filename_prefix}_{i}.json"
                self.file_manager.save_json(chunk_data, str(chunk_file))
                
                chunks.append(chunk_data)
                self.logger.debug(f"Created chunk {i}: {len(chunk_data)} items")
        
        else:
            raise ValueError(f"Unsupported data type for splitting: {type(data)}")
        
        self.logger.info(f"Created {len(chunks)} chunks in {splits_dir}")
        return chunks


class RetryProcessor:
    """Handles retry logic for processing operations."""
    
    def __init__(self, logger: PipelineLogger, max_attempts: int = 3):
        self.logger = logger
        self.max_attempts = max_attempts
        self.retry_items = defaultdict(list)
    
    def execute_with_retry(self, operation, item_id: str, *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts:
                    self.logger.log_retry(attempt, self.max_attempts, str(e))
                    time.sleep(0.1 * attempt)  # Brief delay with backoff
                else:
                    self.retry_items[operation.__name__].append(item_id)
                    self.logger.error(f"Operation failed after {self.max_attempts} attempts for {item_id}: {e}")
        
        return None  # Return None for failed operations
    
    def get_retry_items(self, operation_name: str = None) -> Union[List[str], Dict[str, List[str]]]:
        """Get items that failed and need retry."""
        if operation_name:
            return self.retry_items.get(operation_name, [])
        return dict(self.retry_items)
    
    def save_retry_list(self, filepath: str) -> None:
        """Save retry items to file."""
        retry_data = dict(self.retry_items)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(retry_data, f, indent=4)
        
        self.logger.info(f"Saved retry list to {filepath}")


class ProcessingPipeline:
    """Coordinates execution of multiple processing stages."""
    
    def __init__(self, config: ConfigManager, logger: PipelineLogger):
        self.config = config
        self.logger = logger
        self.stages = []
        self.results = {}
    
    def add_stage(self, processor: BaseProcessor, stage_config: Dict = None) -> None:
        """Add a processing stage to the pipeline."""
        self.stages.append({
            'processor': processor,
            'config': stage_config or {}
        })
        self.logger.info(f"Added stage: {processor.stage_name}")
    
    def run_pipeline(self, start_stage: int = 0, end_stage: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the processing pipeline.
        
        Args:
            start_stage: Index of stage to start from
            end_stage: Index of stage to end at (inclusive)
        
        Returns:
            Dictionary of results from each stage
        """
        end_stage = end_stage or len(self.stages) - 1
        
        self.logger.log_processing_start("PIPELINE", f"Running stages {start_stage}-{end_stage}")
        
        for i in range(start_stage, min(end_stage + 1, len(self.stages))):
            stage = self.stages[i]
            processor = stage['processor']
            stage_config = stage['config']
            
            try:
                self.logger.info(f"Executing stage {i}: {processor.stage_name}")
                result = processor.run(**stage_config)
                self.results[processor.stage_name] = result
                
            except Exception as e:
                self.logger.error(f"Stage {i} ({processor.stage_name}) failed: {e}")
                if not stage_config.get('continue_on_error', False):
                    raise
                else:
                    self.results[processor.stage_name] = None
        
        self.logger.log_processing_end("PIPELINE", f"Completed stages {start_stage}-{end_stage}")
        return self.results
    
    def get_stage_result(self, stage_name: str) -> Any:
        """Get result from a specific stage."""
        return self.results.get(stage_name)


if __name__ == "__main__":
    # Example usage
    from config import ConfigManager
    from logging_utils import get_logger
    
    config = ConfigManager()
    logger = get_logger("base_processor_test")
    
    # Test file manager
    file_manager = FileManager(logger)
    print("Base processor utilities test completed")
