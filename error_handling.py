"""
Robust error handling utilities for the medical data pipeline.
"""
import sys
import traceback
from typing import Optional, Callable, Any, Dict, List
from functools import wraps
from datetime import datetime
from collections import defaultdict
from logging_utils import PipelineLogger


class PipelineError(Exception):
    """Base exception class for pipeline-specific errors."""
    
    def __init__(self, message: str, stage: Optional[str] = None, details: Optional[Dict] = None):
        self.message = message
        self.stage = stage
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        error_info = [self.message]
        if self.stage:
            error_info.append(f"Stage: {self.stage}")
        if self.details:
            details_str = ", ".join([f"{k}={v}" for k, v in self.details.items()])
            error_info.append(f"Details: {details_str}")
        return " | ".join(error_info)


class DataProcessingError(PipelineError):
    """Exception for data processing errors."""
    pass


class ModelError(PipelineError):
    """Exception for model-related errors."""
    pass


class FileOperationError(PipelineError):
    """Exception for file operation errors."""
    pass


class ConfigurationError(PipelineError):
    """Exception for configuration errors."""
    pass


class ValidationError(PipelineError):
    """Exception for data validation errors."""
    pass


class ErrorCollector:
    """Collects and manages errors during pipeline execution."""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.errors: Dict[str, List[Dict]] = defaultdict(list)
        self.warnings: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_error(self, stage: str, error: Exception, context: Optional[Dict] = None) -> None:
        """Add an error to the collection."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        self.errors[stage].append(error_info)
        self.logger.error(f"Error in {stage}: {error}")
    
    def add_warning(self, stage: str, message: str, context: Optional[Dict] = None) -> None:
        """Add a warning to the collection."""
        warning_info = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'context': context or {}
        }
        self.warnings[stage].append(warning_info)
        self.logger.warning(f"Warning in {stage}: {message}")
    
    def has_errors(self, stage: Optional[str] = None) -> bool:
        """Check if there are any errors."""
        if stage:
            return len(self.errors.get(stage, [])) > 0
        return len(self.errors) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors and warnings."""
        return {
            'total_errors': sum(len(errors) for errors in self.errors.values()),
            'total_warnings': sum(len(warnings) for warnings in self.warnings.values()),
            'errors_by_stage': {stage: len(errors) for stage, errors in self.errors.items()},
            'warnings_by_stage': {stage: len(warnings) for stage, warnings in self.warnings.items()},
            'errors': dict(self.errors),
            'warnings': dict(self.warnings)
        }
    
    def save_error_report(self, filepath: str) -> None:
        """Save error report to file."""
        import json
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_error_summary(), f, indent=2)
            self.logger.log_file_operation("save error report", filepath, True)
        except Exception as e:
            self.logger.log_file_operation("save error report", filepath, False)
            raise FileOperationError(f"Failed to save error report: {e}")


def handle_errors(error_collector: Optional[ErrorCollector] = None, 
                 stage: Optional[str] = None,
                 continue_on_error: bool = False):
    """
    Decorator for handling errors in pipeline functions.
    
    Args:
        error_collector: ErrorCollector instance to collect errors
        stage: Name of the pipeline stage
        continue_on_error: Whether to continue execution on error
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_collector and stage:
                    error_collector.add_error(stage, e, {'function': func.__name__})
                
                if continue_on_error:
                    return None
                else:
                    raise
        return wrapper
    return decorator


def validate_inputs(**validators) -> Callable:
    """
    Decorator for validating function inputs.
    
    Args:
        **validators: Dictionary of parameter_name -> validation_function pairs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise ValidationError(
                                f"Validation failed for parameter '{param_name}' in function '{func.__name__}'"
                            )
                    except Exception as e:
                        raise ValidationError(
                            f"Error validating parameter '{param_name}' in function '{func.__name__}': {e}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_file_operation(operation: str):
    """
    Decorator for safe file operations.
    
    Args:
        operation: Description of the file operation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                return result
            except FileNotFoundError as e:
                raise FileOperationError(f"File not found during {operation}: {e}")
            except PermissionError as e:
                raise FileOperationError(f"Permission denied during {operation}: {e}")
            except IOError as e:
                raise FileOperationError(f"IO error during {operation}: {e}")
            except Exception as e:
                raise FileOperationError(f"Unexpected error during {operation}: {e}")
        return wrapper
    return decorator


def graceful_shutdown(logger: PipelineLogger, error_collector: ErrorCollector):
    """
    Handle graceful shutdown of the pipeline.
    
    Args:
        logger: Logger instance
        error_collector: Error collector instance
    """
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        
        # Save error report
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_report_path = f"error_report_{timestamp}.json"
            error_collector.save_error_report(error_report_path)
            logger.info(f"Error report saved to {error_report_path}")
        except Exception as e:
            logger.error(f"Failed to save error report during shutdown: {e}")
        
        # Exit gracefully
        logger.info("Pipeline shutdown completed")
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


class ProgressValidator:
    """Validates progress and detects stalled operations."""
    
    def __init__(self, logger: PipelineLogger, timeout_minutes: int = 30):
        self.logger = logger
        self.timeout_minutes = timeout_minutes
        self.last_progress_time = datetime.now()
        self.last_progress_value = 0
    
    def update_progress(self, current_value: int) -> None:
        """Update progress tracking."""
        if current_value > self.last_progress_value:
            self.last_progress_time = datetime.now()
            self.last_progress_value = current_value
    
    def check_timeout(self) -> bool:
        """Check if operation has timed out."""
        time_since_progress = datetime.now() - self.last_progress_time
        if time_since_progress.total_seconds() > (self.timeout_minutes * 60):
            self.logger.warning(f"No progress detected for {self.timeout_minutes} minutes")
            return True
        return False


class ResourceMonitor:
    """Monitor system resources during pipeline execution."""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.initial_memory = None
        self.peak_memory = 0
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        try:
            import psutil
            self.initial_memory = psutil.virtual_memory().used
            self.logger.info(f"Started resource monitoring. Initial memory: {self.initial_memory / (1024**3):.2f} GB")
        except ImportError:
            self.logger.warning("psutil not available, resource monitoring disabled")
    
    def log_resource_usage(self, stage: str) -> None:
        """Log current resource usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            current_memory = memory.used
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            self.logger.info(f"Resource usage at {stage}:")
            self.logger.info(f"  Memory: {current_memory / (1024**3):.2f} GB ({memory.percent:.1f}%)")
            self.logger.info(f"  CPU: {cpu_percent:.1f}%")
            
            if self.initial_memory:
                memory_increase = current_memory - self.initial_memory
                self.logger.info(f"  Memory increase: {memory_increase / (1024**3):.2f} GB")
            
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"Failed to log resource usage: {e}")
    
    def check_memory_threshold(self, threshold_percent: float = 90.0) -> bool:
        """Check if memory usage exceeds threshold."""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > threshold_percent:
                self.logger.warning(f"Memory usage ({memory_percent:.1f}%) exceeds threshold ({threshold_percent}%)")
                return True
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"Failed to check memory threshold: {e}")
        
        return False


# Validation functions for common data types
def validate_file_exists(filepath: str) -> bool:
    """Validate that a file exists."""
    from pathlib import Path
    return Path(filepath).exists()


def validate_directory_exists(dirpath: str) -> bool:
    """Validate that a directory exists."""
    from pathlib import Path
    return Path(dirpath).is_dir()


def validate_positive_integer(value: Any) -> bool:
    """Validate that a value is a positive integer."""
    return isinstance(value, int) and value > 0


def validate_float_range(min_val: float = None, max_val: float = None):
    """Create a validator for float values in a specific range."""
    def validator(value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    return validator


def validate_non_empty_string(value: Any) -> bool:
    """Validate that a value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def validate_list_not_empty(value: Any) -> bool:
    """Validate that a value is a non-empty list."""
    return isinstance(value, list) and len(value) > 0


if __name__ == "__main__":
    # Example usage
    from logging_utils import get_logger
    
    logger = get_logger("error_handling_test")
    error_collector = ErrorCollector(logger)
    
    # Test error collection
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_collector.add_error("test_stage", e, {"test_context": "example"})
    
    # Test validation
    @validate_inputs(
        filepath=validate_file_exists,
        count=validate_positive_integer
    )
    def test_function(filepath: str, count: int):
        return f"Processing {count} items from {filepath}"
    
    # Test resource monitoring
    monitor = ResourceMonitor(logger)
    monitor.start_monitoring()
    monitor.log_resource_usage("test")
    
    print("Error handling utilities test completed")
    print(f"Error summary: {error_collector.get_error_summary()}")