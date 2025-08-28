"""
Azure integration utilities for the medical data pipeline.
Preserves all Azure-specific authentication and workspace configurations.
"""
import os
import requests
from typing import Optional
from azureml.core import Workspace
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
from config import ConfigManager, ModelConfig
from logging_utils import PipelineLogger


class BearerAuth(requests.auth.AuthBase):
    """Bearer token authentication for Azure requests."""
    
    def __init__(self, token: str):
        self.token = token
    
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class AzureModelManager:
    """Manages Azure OpenAI model initialization and configuration."""
    
    def __init__(self, config: ConfigManager, logger: PipelineLogger):
        self.config = config
        self.logger = logger
        self.workspace = None
        self.keyvault = None
        self.credential = None
        self._initialize_azure_resources()
    
    def _initialize_azure_resources(self) -> None:
        """Initialize Azure workspace and credentials."""
        try:
            self.workspace = Workspace.from_config()
            self.keyvault = self.workspace.get_default_keyvault()
            self.credential = DefaultAzureCredential()
            self.logger.info("Azure resources initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure resources: {e}")
            raise
    
    def _setup_azure_environment(self) -> None:
        """Setup Azure environment variables."""
        try:
            workspacename = self.keyvault.get_secret("project-workspace-name")
            access_token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
            subscription_id = self.keyvault.get_secret("project-subscription-id")
            
            os.environ["AZURE_OPENAI_KEY"] = access_token.token
            os.environ["AZURE_OPENAI_ENDPOINT"] = f"https://{workspacename}openai.openai.azure.com/"
            os.environ["AZURE_OPENAI_API_KEY"] = "ee0dd46654bd4427ba4f5580b5a0db0a"
            os.environ["AZURE_OPENAI_API_BASE"] = "https://xqrojjmb2wjlqopopenai.openai.azure.com/"
            
            self.logger.debug("Azure environment variables configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Azure environment: {e}")
            raise
    
    def initialize_model(self, model_type: str = "gpt-4o") -> AzureChatOpenAI:
        """
        Initialize Azure OpenAI model.
        
        Args:
            model_type: Type of model to initialize ('gpt-4o', 'gpt-4.1', 'o4-mini')
        
        Returns:
            Initialized AzureChatOpenAI model
        """
        self._setup_azure_environment()
        
        model_configs = {
            "gpt-4o": {
                "version": self.config.model.gpt_4o_version,
                "deployment": self.config.model.gpt_4o_deployment,
                "api_version": "2023-10-01-preview"
            },
            "gpt-4.1": {
                "version": self.config.model.gpt_41_version,
                "deployment": self.config.model.gpt_41_deployment,
                "api_version": "2024-12-01-preview"
            },
            "o4-mini": {
                "version": self.config.model.o4_mini_version,
                "deployment": self.config.model.o4_mini_deployment,
                "api_version": "2024-12-01-preview"
            }
        }
        
        if model_type not in model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_config = model_configs[model_type]
        
        try:
            # Set environment variables for the specific model
            os.environ["AZURE_OPENAI_API_VERSION"] = model_config["version"]
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = model_config["deployment"]
            
            # Setup deployment URL
            workspacename = self.keyvault.get_secret("project-workspace-name")
            subscription_id = self.keyvault.get_secret("project-subscription-id")
            api_version = model_config["api_version"]
            
            url = (f"https://management.azure.com/subscriptions/{subscription_id}/"
                   f"resourceGroups/{workspacename}-common/providers/"
                   f"Microsoft.CognitiveServices/accounts/{workspacename}openai/"
                   f"deployments?api-version={api_version}")
            
            access_token = self.credential.get_token("https://management.azure.com/.default")
            response = requests.get(url, auth=BearerAuth(access_token.token))
            
            self.logger.log_model_operation(model_config["deployment"], "initialization", True)
            
            # Initialize the model
            model = AzureChatOpenAI(
                deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                model_kwargs={"seed": self.config.model.seed}
            )
            
            self.logger.info(f"Model {model_type} ({model_config['deployment']}) initialized successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model {model_type}: {e}")
            self.logger.log_model_operation(model_config["deployment"], "initialization", False)
            raise


class RetryManager:
    """Manages retry logic for API calls and operations."""
    
    def __init__(self, logger: PipelineLogger, max_attempts: int = 3, delay: float = 0.1):
        self.logger = logger
        self.max_attempts = max_attempts
        self.delay = delay
    
    def execute_with_retry(self, operation, *args, **kwargs):
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
        
        Returns:
            Result of the operation
        """
        import time
        
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts:
                    self.logger.log_retry(attempt, self.max_attempts, str(e))
                    time.sleep(self.delay * attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Operation failed after {self.max_attempts} attempts: {e}")
        
        raise last_exception


def create_azure_manager(config: ConfigManager, logger: PipelineLogger) -> AzureModelManager:
    """
    Factory function to create AzureModelManager.
    
    Args:
        config: Configuration manager instance
        logger: Logger instance
    
    Returns:
        AzureModelManager instance
    """
    return AzureModelManager(config, logger)


# Legacy function for backward compatibility
def initialize_llm(model_name: str) -> AzureChatOpenAI:
    """
    Legacy function for initializing Azure OpenAI models.
    Maintained for backward compatibility with existing code.
    
    Args:
        model_name: Name of the model to initialize
    
    Returns:
        Initialized AzureChatOpenAI model
    """
    from config import ConfigManager
    from logging_utils import get_logger
    
    config = ConfigManager()
    logger = get_logger("azure_legacy")
    azure_manager = AzureModelManager(config, logger)
    
    return azure_manager.initialize_model(model_name)


if __name__ == "__main__":
    # Example usage
    from config import ConfigManager
    from logging_utils import get_logger
    
    config = ConfigManager()
    logger = get_logger("azure_test")
    azure_manager = create_azure_manager(config, logger)
    
    # Test model initialization
    try:
        model = azure_manager.initialize_model("gpt-4o")
        logger.info("Azure model manager test completed successfully")
    except Exception as e:
        logger.error(f"Azure model manager test failed: {e}")
