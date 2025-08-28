"""
Hyperparameter tuning for medical query classification models using Optuna.
"""
import math
import json
import random
import numpy as np
import pandas as pd
import torch
import optuna
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.losses import ContrastiveLoss

from base_processor import BaseProcessor, FileManager
from config import ConfigManager
from logging_utils import PipelineLogger, get_logger
from error_handling import handle_errors, validate_inputs, validate_positive_integer, validate_float_range


class HyperparameterConfig:
    """Configuration for hyperparameter tuning."""
    
    # Parameter ranges for Optuna
    BATCH_SIZE_OPTIONS = [32, 64, 128, 256]
    EPOCHS_RANGE = (1, 10)
    LEARNING_RATE_RANGE = (1e-6, 1e-4)
    MARGIN_RANGE = (0.2, 1.0)
    
    # Model configuration
    DEFAULT_MODEL_NAME = "../model/NeuML_pubmedbert-base-embeddings"
    EVALUATION_STEPS = 1000
    
    # Dataset sampling
    DEFAULT_SAMPLE_FRACTION = 0.011


class DatasetManager:
    """Manages dataset loading and preparation for hyperparameter tuning."""
    
    def __init__(self, file_manager: FileManager, logger: PipelineLogger):
        self.file_manager = file_manager
        self.logger = logger
    
    def load_datasets(self, train_path: str, eval_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and evaluation datasets."""
        try:
            train_df = pd.read_csv(train_path).iloc[:, 1:]  # Remove index column
            eval_df = pd.read_csv(eval_path).iloc[:, 1:]
            
            self.logger.info(f"Loaded training dataset: {len(train_df)} samples")
            self.logger.info(f"Loaded evaluation dataset: {len(eval_df)} samples")
            
            return train_df, eval_df
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise
    
    def create_stratified_sample(self, data: pd.DataFrame, fraction: float) -> pd.DataFrame:
        """Create stratified sample from dataset based on anchor queries."""
        try:
            # Get value counts for anchors
            anchor_counts = data['anchor'].value_counts().reset_index()
            unique_anchors = anchor_counts.shape[0]
            
            # Sample by anchor to maintain distribution
            sampled_data = data.groupby('anchor', group_keys=False).apply(
                lambda x: x.sample(frac=fraction, random_state=42)
            )
            
            # Verify we still have all unique anchors
            sampled_unique_anchors = sampled_data['anchor'].nunique()
            
            if sampled_unique_anchors != unique_anchors:
                self.logger.warning(f"Sample lost some anchors: {unique_anchors} -> {sampled_unique_anchors}")
            
            sampled_data = sampled_data.reset_index(drop=True)
            self.logger.info(f"Created stratified sample: {len(sampled_data)} samples ({fraction*100:.1f}%)")
            
            return sampled_data
            
        except Exception as e:
            self.logger.error(f"Failed to create stratified sample: {e}")
            raise
    
    def convert_to_input_examples(self, data: pd.DataFrame) -> List[InputExample]:
        """Convert DataFrame to SentenceTransformer InputExample format."""
        examples = []
        
        for idx, row in data.iterrows():
            example = InputExample(
                texts=[str(row['sentence1']), str(row['sentence2'])],
                label=float(row['label'])
            )
            examples.append(example)
        
        self.logger.info(f"Converted {len(examples)} samples to InputExample format")
        return examples
    
    def build_huggingface_dataset(self, train_path: str, eval_path: str) -> DatasetDict:
        """Build HuggingFace dataset from CSV files."""
        train_data, eval_data = self.load_datasets(train_path, eval_path)
        
        # Shuffle data
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        eval_data = eval_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Convert to HuggingFace format
        train_dataset = Dataset.from_pandas(train_data)
        eval_dataset = Dataset.from_pandas(eval_data)
        
        dataset_dict = DatasetDict()
        dataset_dict['train'] = train_dataset
        dataset_dict['validation'] = eval_dataset
        
        self.logger.info("Created HuggingFace DatasetDict")
        return dataset_dict


class RelevantDocumentsManager:
    """Manages creation of relevant documents mapping for evaluation."""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
    
    def create_relevant_documents_mapping(self, eval_dataset: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]:
        """Create mapping of queries to relevant documents for evaluation."""
        queries = {}
        corpus = {}
        relevant_docs = defaultdict(set)
        
        query_id_map = {}
        doc_id_map = {}
        query_id_counter = 0
        doc_id_counter = 0
        
        for idx, row in eval_dataset.iterrows():
            sentence1 = row['sentence1']
            sentence2 = row['sentence2']
            label = int(row['label'])
            
            # Assign query IDs
            if sentence1 not in query_id_map:
                query_id = f'q{query_id_counter}'
                query_id_map[sentence1] = query_id
                queries[query_id] = sentence1
                query_id_counter += 1
            else:
                query_id = query_id_map[sentence1]
            
            # Assign document IDs
            if sentence2 not in doc_id_map:
                doc_id = f'd{doc_id_counter}'
                doc_id_map[sentence2] = doc_id
                corpus[doc_id] = sentence2
                doc_id_counter += 1
            else:
                doc_id = doc_id_map[sentence2]
            
            # Map relevant documents
            if label == 1:
                relevant_docs[query_id].add(doc_id)
        
        # Convert to regular dict
        relevant_docs = dict(relevant_docs)
        
        self.logger.info(f"Created relevant documents mapping: {len(queries)} queries, {len(corpus)} documents")
        return queries, corpus, relevant_docs


class RecallAtKEvaluator(SentenceEvaluator):
    """Custom evaluator that computes mean Recall@K."""
    
    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str], 
                 relevant_docs: Dict[str, set], k: int = 3, name: str = ''):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.k = k
        self.name = name
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1) -> float:
        """Compute Recall@K for the model."""
        # Compute embeddings for queries and corpus
        query_ids = list(self.queries.keys())
        corpus_ids = list(self.corpus.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        corpus_texts = [self.corpus[cid] for cid in corpus_ids]
        
        query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=False)
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute similarity scores
        cos_scores = torch.matmul(query_embeddings, corpus_embeddings.T)
        
        # For each query, compute Recall@K
        recall_scores = []
        for idx, query_id in enumerate(query_ids):
            relevant = self.relevant_docs.get(query_id, set())
            if len(relevant) == 0:
                continue  # Skip queries with no relevant documents
            
            scores = cos_scores[idx]
            top_k = torch.topk(scores, k=self.k, largest=True)
            top_k_indices = top_k.indices.cpu().numpy()
            retrieved_doc_ids = [corpus_ids[i] for i in top_k_indices]
            
            num_relevant_retrieved = len(set(retrieved_doc_ids).intersection(relevant))
            recall = num_relevant_retrieved / len(relevant)
            recall_scores.append(recall)
        
        mean_recall = np.mean(recall_scores) if recall_scores else 0.0
        return mean_recall


class HyperparameterOptimizer:
    """Handles Optuna-based hyperparameter optimization."""
    
    def __init__(self, config: HyperparameterConfig, dataset_manager: DatasetManager, 
                 relevant_docs_manager: RelevantDocumentsManager, logger: PipelineLogger):
        self.config = config
        self.dataset_manager = dataset_manager
        self.relevant_docs_manager = relevant_docs_manager
        self.logger = logger
        self.train_dataset = None
        self.dev_evaluator = None
    
    def setup_datasets(self, train_path: str, eval_path: str, sample_fraction: Optional[float] = None):
        """Setup training and evaluation datasets."""
        # Load datasets
        train_df, eval_df = self.dataset_manager.load_datasets(train_path, eval_path)
        
        # Create samples if requested
        if sample_fraction:
            train_df = self.dataset_manager.create_stratified_sample(train_df, sample_fraction)
            eval_df = self.dataset_manager.create_stratified_sample(eval_df, sample_fraction)
        
        # Convert to InputExample format
        self.train_dataset = self.dataset_manager.convert_to_input_examples(train_df)
        
        # Create evaluator
        queries, corpus, relevant_docs = self.relevant_docs_manager.create_relevant_documents_mapping(eval_df)
        self.dev_evaluator = RecallAtKEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            k=3,
            name="recall_at_3_evaluator"
        )
        
        self.logger.info("Datasets and evaluator setup completed")
    
    def objective(self, trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        try:
            # Suggest hyperparameters
            batch_size = trial.suggest_categorical('batch_size', self.config.BATCH_SIZE_OPTIONS)
            num_epochs = trial.suggest_int('num_epochs', *self.config.EPOCHS_RANGE)
            learning_rate = trial.suggest_float("learning_rate", *self.config.LEARNING_RATE_RANGE, log=True)
            margin = trial.suggest_float('margin', *self.config.MARGIN_RANGE)
            
            self.logger.info(f"Trial {trial.number}: batch_size={batch_size}, epochs={num_epochs}, "
                           f"lr={learning_rate:.2e}, margin={margin:.3f}")
            
            # Create data loader
            train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
            
            # Initialize model
            model = SentenceTransformer(self.config.DEFAULT_MODEL_NAME)
            
            # Define loss function
            train_loss = ContrastiveLoss(model=model, margin=margin)
            
            # Train model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=self.dev_evaluator,
                epochs=num_epochs,
                optimizer_params={'lr': learning_rate},
                show_progress_bar=False,
                evaluation_steps=self.config.EVALUATION_STEPS,
                output_path=None
            )
            
            # Evaluate model
            recall_at_3 = self.dev_evaluator(model)
            
            self.logger.info(f"Trial {trial.number} completed: Recall@3 = {recall_at_3:.4f}")
            return recall_at_3
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst possible score for failed trials
    
    @validate_inputs(
        n_trials=validate_positive_integer,
        timeout=lambda x: x is None or validate_positive_integer(x)
    )
    def optimize(self, n_trials: int = 10, timeout: Optional[int] = None, 
                study_name: Optional[str] = None) -> optuna.Study:
        """Run hyperparameter optimization."""
        # Create study
        if study_name is None:
            study_name = f"medical_query_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            direction='maximize',  # Maximize Recall@3
            study_name=study_name
        )
        
        self.logger.info(f"Starting optimization: {n_trials} trials, timeout={timeout}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Log results
        self.logger.info("Optimization completed!")
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best Recall@3: {study.best_value:.4f}")
        self.logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            self.logger.info(f"  {key}: {value}")
        
        return study


class HyperparameterTuningProcessor(BaseProcessor):
    """Main processor for hyperparameter tuning."""
    
    def __init__(self, config: ConfigManager, logger: PipelineLogger):
        super().__init__(config, logger, "HYPERPARAMETER_TUNING")
        self.file_manager = FileManager(logger)
        self.hp_config = HyperparameterConfig()
        self.dataset_manager = DatasetManager(self.file_manager, logger)
        self.relevant_docs_manager = RelevantDocumentsManager(logger)
        self.optimizer = HyperparameterOptimizer(
            self.hp_config, self.dataset_manager, self.relevant_docs_manager, logger
        )
    
    def create_hyperparameter_samples(self, train_path: str, eval_path: str, 
                                    sample_train_path: str, sample_eval_path: str,
                                    sample_fraction: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified samples for hyperparameter tuning."""
        if sample_fraction is None:
            sample_fraction = self.hp_config.DEFAULT_SAMPLE_FRACTION
        
        # Load original datasets
        train_df, eval_df = self.dataset_manager.load_datasets(train_path, eval_path)
        
        # Create stratified samples
        train_sample = self.dataset_manager.create_stratified_sample(train_df, sample_fraction)
        eval_sample = self.dataset_manager.create_stratified_sample(eval_df, sample_fraction)
        
        # Save samples
        Path(sample_train_path).parent.mkdir(parents=True, exist_ok=True)
        Path(sample_eval_path).parent.mkdir(parents=True, exist_ok=True)
        
        train_sample.to_csv(sample_train_path, index=False)
        eval_sample.to_csv(sample_eval_path, index=False)
        
        self.logger.info(f"Saved training sample: {sample_train_path}")
        self.logger.info(f"Saved evaluation sample: {sample_eval_path}")
        
        return train_sample, eval_sample
    
    def save_optimization_results(self, study: optuna.Study, output_path: str) -> None:
        """Save optimization results to file."""
        results = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_trial.params,
                'datetime_start': study.best_trial.datetime_start.isoformat() if study.best_trial.datetime_start else None,
                'datetime_complete': study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None
            },
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                }
                for trial in study.trials
            ]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.file_manager.save_json(results, output_path)
        self.logger.info(f"Saved optimization results to {output_path}")
    
    @validate_inputs(
        n_trials=validate_positive_integer,
        sample_fraction=validate_float_range(0.001, 1.0)
    )
    def process(self,
                train_path: str,
                eval_path: str,
                n_trials: int = 10,
                timeout: Optional[int] = None,
                sample_fraction: Optional[float] = None,
                output_dir: str = "hyperparameter_results",
                study_name: Optional[str] = None,
                create_samples: bool = True) -> Dict[str, Any]:
        """
        Process hyperparameter tuning.
        
        Args:
            train_path: Path to training dataset
            eval_path: Path to evaluation dataset
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            sample_fraction: Fraction of data to sample for tuning
            output_dir: Directory to save results
            study_name: Name for the optimization study
            create_samples: Whether to create sample datasets
        
        Returns:
            Dictionary containing optimization results
        """
        results = {}
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create samples if requested
        if create_samples:
            sample_train_path = str(output_path / "hyperparameter_samples_training.csv")
            sample_eval_path = str(output_path / "hyperparameter_samples_eval.csv")
            
            train_sample, eval_sample = self.create_hyperparameter_samples(
                train_path, eval_path, sample_train_path, sample_eval_path, sample_fraction
            )
            
            # Use sample paths for optimization
            train_path = sample_train_path
            eval_path = sample_eval_path
            
            results['sample_stats'] = {
                'train_samples': len(train_sample),
                'eval_samples': len(eval_sample),
                'sample_fraction': sample_fraction or self.hp_config.DEFAULT_SAMPLE_FRACTION
            }
        
        # Setup datasets for optimization
        self.optimizer.setup_datasets(train_path, eval_path, sample_fraction)
        
        # Run optimization
        study = self.optimizer.optimize(
            n_trials=n_trials,
            timeout=timeout,
            study_name=study_name
        )
        
        # Save results
        results_file = str(output_path / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.save_optimization_results(study, results_file)
        
        # Create HuggingFace dataset with best parameters (optional)
        hf_dataset = self.dataset_manager.build_huggingface_dataset(train_path, eval_path)
        hf_output_path = output_path / "huggingface_dataset"
        hf_dataset.save_to_disk(str(hf_output_path))
        
        results.update({
            'study': study,
            'best_params': study.best_trial.params,
            'best_score': study.best_value,
            'n_trials_completed': len(study.trials),
            'results_file': results_file,
            'hf_dataset_path': str(hf_output_path)
        })
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for medical query classification")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")
    parser.add_argument("--train-path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--eval-path", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--sample-fraction", type=float, help="Fraction of data to sample")
    parser.add_argument("--output-dir", type=str, default="hyperparameter_results", help="Output directory")
    parser.add_argument("--study-name", type=str, help="Name for the optimization study")
    parser.add_argument("--no-samples", action="store_true", help="Skip creating sample datasets")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup configuration and logging
    config = ConfigManager(args.config)
    logger = get_logger("hyperparameter_tuner", args.log_level, log_dir="logs")
    
    # Create and run processor
    processor = HyperparameterTuningProcessor(config, logger)
    
    try:
        results = processor.run(
            train_path=args.train_path,
            eval_path=args.eval_path,
            n_trials=args.n_trials,
            timeout=args.timeout,
            sample_fraction=args.sample_fraction,
            output_dir=args.output_dir,
            study_name=args.study_name,
            create_samples=not args.no_samples
        )
        
        logger.info("Hyperparameter tuning completed successfully")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best Recall@3: {results['best_score']:.4f}")
        logger.info(f"Results saved to: {results['results_file']}")
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise


if __name__ == '__main__':
    main()