"""
Standardized logging utilities for the medical data pipeline.
"""
import os
import sys
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path


class PipelineLogger:
    """Standardized logger for the medical data pipeline."""
    
    def __init__(self, 
                 name: str,
                 log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 console_output: bool = True):
        """
        Initialize the logger.
        
        Args:
            name: Logger name (usually the module/script name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to save log files (optional)
            console_output: Whether to output logs to console
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir_path / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_processing_start(self, stage: str, description: str) -> None:
        """Log the start of a processing stage."""
        self.info(f"="*60)
        self.info(f"STARTING {stage}: {description}")
        self.info(f"="*60)
    
    def log_processing_end(self, stage: str, description: str) -> None:
        """Log the end of a processing stage."""
        self.info(f"="*60)
        self.info(f"COMPLETED {stage}: {description}")
        self.info(f"="*60)
    
    def log_progress(self, current: int, total: int, item: str = "items") -> None:
        """Log progress information."""
        percentage = (current / total) * 100 if total > 0 else 0
        self.info(f"Progress: {current}/{total} {item} ({percentage:.1f}%)")
    
    def log_retry(self, attempt: int, max_attempts: int, error: str) -> None:
        """Log retry attempts."""
        self.warning(f"Retry attempt {attempt}/{max_attempts} due to error: {error}")
    
    def log_file_operation(self, operation: str, filepath: str, success: bool = True) -> None:
        """Log file operations."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"File {operation} {status}: {filepath}")
    
    def log_model_operation(self, model_name: str, operation: str, success: bool = True) -> None:
        """Log model operations."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Model {operation} {status}: {model_name}")
    
    def log_data_stats(self, description: str, stats: dict) -> None:
        """Log data statistics."""
        self.info(f"Data Statistics - {description}:")
        for key, value in stats.items():
            self.info(f"  {key}: {value}")


def get_logger(name: str, 
               log_level: str = "INFO", 
               log_dir: Optional[str] = None,
               console_output: bool = True) -> PipelineLogger:
    """
    Get a standardized logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_dir: Directory for log files
        console_output: Whether to output to console
    
    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(name, log_level, log_dir, console_output)


def setup_default_logging() -> None:
    """Setup default logging configuration for the pipeline."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, logger: PipelineLogger, total_items: int, description: str = "items"):
        self.logger = logger
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current_item += increment
        if self.current_item % max(1, self.total_items // 20) == 0:  # Log every 5%
            self.logger.log_progress(self.current_item, self.total_items, self.description)
    
    def complete(self) -> None:
        """Mark as complete and log timing."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Completed processing {self.total_items} {self.description} in {duration}")


if __name__ == "__main__":
    # Example usage
    logger = get_logger("test_logger", log_dir="logs")
    logger.info("This is a test log message")
    logger.log_processing_start("STAGE1", "Testing logging functionality")
    logger.log_data_stats("Test Data", {"records": 1000, "features": 50})
    logger.log_processing_end("STAGE1", "Testing logging functionality")
