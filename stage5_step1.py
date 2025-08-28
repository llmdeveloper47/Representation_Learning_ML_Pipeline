# python3 stage_5_step1.py

import os
import json
import gzip
import random
import joblib

import torch
import pickle
import logging

import numpy as np
import pandas as pd
import faiss
import logging

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Union, List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Annotated_DataLoader:
    def __init__(self, file_type : str, directory_path : str):
        self.file_type = file_type
        self.directory_path = directory_path

    def process_codes(self, data : dict):
        """ parses out icd code description from the code : description string.
        """

        # get the specialty
        specialt_query_code_description_dict = {}
        specialty = list(data.keys())[0]

        query_code_description_dict = list(data.values())[0]
        query_code_description_diltered_dict = {}
        for query, icd_code_descriptions in query_code_description_dict.items():
            code_description_list = []
            for description in icd_code_descriptions:
                # the description is as follows ->  code : some text description
                # from the description , parse out the path after the : and then strip it
                # and then attach the description in a list
                code_description_list.append(str(description.split(':')[1]).strip())
            # once all the descriptions for a query are processed, add them to a dictionary
            query_code_description_diltered_dict[query] = code_description_list

        # once all the queries for a specialty are processed, add the processed result to a dictionary
        specialt_query_code_description_dict[specialty] = query_code_description_diltered_dict

        return specialt_query_code_description_dict
    
    
    def filter_dataset(self, data_list):
        """ filters out all ICD code descrptions which are empty
        """
        filtered_data_list = []
        for i in range(len(data_list)):
            data_specialty = data_list[i]
    
            specialty = list(data_specialty.keys())[0]
            query_codes = data_specialty.get(specialty)
            query_code_filtered = {}
    
    
            for query, code_descriptions in query_codes.items():
                if len(code_descriptions) != 1:
                    code_descriptions_filtered = [code for code in code_descriptions if code != ''] 
                    query_code_filtered[query] = code_descriptions_filtered
                else:
                    continue
    
            data_specialty_filtered = {}
            data_specialty_filtered[specialty] = query_code_filtered
            filtered_data_list.append(data_specialty_filtered)
            
        return filtered_data_list

    def get_all_queries(self, data_list : list):
        """ gets a list of all queries used across the dataset
        """
        all_queries = []
    
        for i in tqdm(range(len(data_list))):
    
            specialty_data = data_list[i]
    
            specialty = list(specialty_data.keys())[0]
            #print(f'Processing Specialty : {specialty}')
    
            query_positives_dataset = specialty_data.get(specialty)
            all_queries = all_queries + list(query_positives_dataset.keys())
            
        return list(set(all_queries))  


    def load_annotated_files_gpt_41(self):
        """ loads the annotated dataset ( using GPT 4o and GPT 4.1 ) and evaluated dataset using gpt4.1
            and return a tuple of filtered_data_list and all_queries
        """
        
        if self.file_type == '.json':
    
            path = self.directory_path
    
            all_files = [path + file for file in os.listdir(path) if '.json' in file]
            data_list = []
            for file_path in all_files:
    
                with open(file_path, 'r') as file:
                    data = json.load(file)

                data_processed = self.process_codes(data = data)

                data_list.append(data_processed)
        
        elif self.file_type == '.pkl':
            
            with open(self.directory_path, 'rb') as file:
                data_list = pickle.load(file)
        print(f'Total Files : {len(data_list)}')

        filtered_data_list =  self.filter_dataset(data_list = data_list)
        all_queries = self.get_all_queries(data_list = filtered_data_list)
        print(f'Total Dataset Size : {len(filtered_data_list)}')
        print(f'Total Queries : {len(all_queries)}')

        return filtered_data_list, all_queries


class ICD_DataLoader:
    def __init__(self, icd_reference_file : str):
        self.icd_reference_file = icd_reference_file

    def load_icd_files(self):
        
        dataset_icd = pd.read_csv(icd_reference_file).iloc[:,1:]
        
        dataset_icd = dataset_icd.drop_duplicates()
        dataset_icd = dataset_icd.iloc[:,13:15]
        dataset_icd.columns = ['ICD_Codes','Description']
        dataset_icd['ICD_Codes'] = dataset_icd['ICD_Codes'].apply(lambda x : x.strip())
        dataset_icd['Description'] = dataset_icd['Description'].apply(lambda x : x.strip())
        dataset_icd = dataset_icd.drop_duplicates(subset = ['ICD_Codes'], keep = 'first')
        dataset_icd = dataset_icd.reset_index()
        
        dataset_icd = dataset_icd[['ICD_Codes', 'Description']]
        dataset_icd.columns = ['codes','description']
        
        icd_reference_lookup = {}
        for row in dataset_icd.itertuples():
            icd_reference_lookup[row.codes.strip()] = row.description.strip()

        return icd_reference_lookup       

class Distributed_Sentence_Embeddings:
    """ Generates distributed embedding for descriptions
        Args:
            model_path : path to the Sentence Transformer Model
            batch_size : batch size to be used for building the index of code : embedding pairs
            text_descriptions : all anchor positive negative text
        Returns:
            text_sentence_embedding_dictionary : dictionary with all sentences ( anchor + positives + negatives ) and respective embedding
    """
    
    def __init__(self, model_path : str, batch_size : int):
        """ initialize model_path and reference code_lookup dataset
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.text_sentence_embedding_dictionary : dict =  {}
        self.model = None
    def load_model(self):
        """ loads SentenceTransformer model
        """
        self.model = SentenceTransformer(model_path, trust_remote_code=True).cuda()
        return self.model
        
    def setup_multi_processing_pool(self):
        """ determines the number of GPUs available on the compute
        """
        
        print("Setting Up Multi Processing Pool")
        device_count = torch.cuda.device_count()
        pool_list = [f"cuda:{device}" for device in range(device_count)]
        return pool_list
    
    def generate_sentence_embedding_pair(self, text_descriptions : List[str]):
        """ Creates a dictionary of code and respective embeddings
        """
        
        print("Generating Embeddings")

        torch.cuda.empty_cache()

        model = self.load_model()

        pool_list = self.setup_multi_processing_pool()

        pool = self.model.start_multi_process_pool(pool_list)

        print("Computing embeddings using the multi-process pool")
        embeddings = self.model.encode_multi_process(sentences = text_descriptions, pool = pool, batch_size = self.batch_size)

        self.model.stop_multi_process_pool(pool)
        
        for sentence, embedding in zip(text_descriptions, embeddings):
            self.text_sentence_embedding_dictionary[sentence] = embedding

        print("Sentence : Embedding Mapping Created")
        return self.text_sentence_embedding_dictionary

class ICDEmbeddingsIO:
    """Efficient saving and loading of ICD code embeddings dictionary."""
    
    @staticmethod
    def save_pickle(embeddings_dict: Dict[str, np.ndarray], filepath: str, compress: bool = True) -> None:
        """
        Save embeddings dictionary using pickle with optional compression.
        
        Args:
            embeddings_dict: Dictionary with ICD descriptions as keys and embeddings as values
            filepath: Path to save the file
            compress: Whether to use gzip compression (recommended for large files)
        """
        filepath = Path(filepath)
        
        try:
            if compress:
                with gzip.open(f"{filepath}.pkl.gz", 'wb') as f:
                    pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Compressed embeddings saved to {filepath}.pkl.gz")
            else:
                with open(f"{filepath}.pkl", 'wb') as f:
                    pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Embeddings saved to {filepath}.pkl")
                
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    @staticmethod
    def load_pickle(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings dictionary from pickle file.
        
        Args:
            filepath: Path to the saved file (with or without extension)
            
        Returns:
            Dictionary with ICD descriptions as keys and embeddings as values
        """
        filepath = Path(filepath)
        
        # Try compressed file first
        compressed_path = filepath.with_suffix('.pkl.gz') if filepath.suffix != '.gz' else filepath
        regular_path = filepath.with_suffix('.pkl') if filepath.suffix != '.pkl' else filepath
        
        try:
            if compressed_path.exists():
                with gzip.open(compressed_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded compressed embeddings from {compressed_path}")
                return embeddings_dict
            elif regular_path.exists():
                with open(regular_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded embeddings from {regular_path}")
                return embeddings_dict
            else:
                raise FileNotFoundError(f"No embeddings file found at {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    @staticmethod
    def save_joblib(embeddings_dict: Dict[str, np.ndarray], filepath: str, compress: int = 3) -> None:
        """
        Save embeddings dictionary using joblib (good for numpy arrays).
        
        Args:
            embeddings_dict: Dictionary with ICD descriptions as keys and embeddings as values
            filepath: Path to save the file
            compress: Compression level (0-9, 3 is good balance of speed/size)
        """
        filepath = Path(filepath).with_suffix('.joblib')
        
        try:
            joblib.dump(embeddings_dict, filepath, compress=compress)
            logger.info(f"Embeddings saved with joblib to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings with joblib: {e}")
            raise
    
    @staticmethod
    def load_joblib(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings dictionary from joblib file.
        
        Args:
            filepath: Path to the saved file
            
        Returns:
            Dictionary with ICD descriptions as keys and embeddings as values
        """
        filepath = Path(filepath).with_suffix('.joblib')
        
        try:
            embeddings_dict = joblib.load(filepath)
            logger.info(f"Loaded embeddings with joblib from {filepath}")
            return embeddings_dict
        except Exception as e:
            logger.error(f"Error loading embeddings with joblib: {e}")
            raise
    
    @staticmethod
    def save_npz(embeddings_dict: Dict[str, np.ndarray], filepath: str) -> None:
        """
        Save embeddings using numpy's compressed format.
        Efficient for large numpy arrays but keys must be valid Python identifiers.
        
        Args:
            embeddings_dict: Dictionary with ICD descriptions as keys and embeddings as values
            filepath: Path to save the file
        """
        filepath = Path(filepath).with_suffix('.npz')
        
        try:
            # Convert keys to valid numpy savez keys and save mapping
            key_mapping = {f"embed_{i}": key for i, key in enumerate(embeddings_dict.keys())}
            reverse_mapping = {v: k for k, v in key_mapping.items()}
            
            # Prepare data for numpy
            np_dict = {reverse_mapping[key]: value for key, value in embeddings_dict.items()}
            
            # Save both embeddings and key mapping
            np.savez_compressed(filepath, key_mapping=np.array(list(key_mapping.items()), dtype=object), **np_dict)
            logger.info(f"Embeddings saved with numpy to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings with numpy: {e}")
            raise
    
    @staticmethod
    def load_npz(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings dictionary from numpy compressed file.
        
        Args:
            filepath: Path to the saved file
            
        Returns:
            Dictionary with ICD descriptions as keys and embeddings as values
        """
        filepath = Path(filepath).with_suffix('.npz')
        
        try:
            data = np.load(filepath, allow_pickle=True)
            key_mapping = dict(data['key_mapping'])
            
            embeddings_dict = {}
            for np_key, original_key in key_mapping.items():
                embeddings_dict[original_key] = data[np_key]
            
            logger.info(f"Loaded embeddings with numpy from {filepath}")
            return embeddings_dict
            
        except Exception as e:
            logger.error(f"Error loading embeddings with numpy: {e}")
            raise
            
# Convenience functions for quick usage
def save_embeddings(embeddings_dict: Dict[str, np.ndarray], filepath: str, method: str = 'pickle') -> None:
    """
    Quick save function with method selection.
    
    Args:
        embeddings_dict: Dictionary to save
        filepath: Path to save file
        method: 'pickle', 'joblib', or 'npz'
    """
    io_handler = ICDEmbeddingsIO()
    
    if method == 'pickle':
        io_handler.save_pickle(embeddings_dict, filepath, compress=True)
    elif method == 'joblib':
        io_handler.save_joblib(embeddings_dict, filepath)
    elif method == 'npz':
        io_handler.save_npz(embeddings_dict, filepath)
    else:
        raise ValueError("Method must be 'pickle', 'joblib', or 'npz'")


def load_embeddings(file_path : str):
    
    @staticmethod
    def load_pickle(filepath: str) -> Dict[str, np.ndarray]:

        filepath = Path(filepath)
        # Try compressed file first
        compressed_path = filepath.with_suffix('.pkl.gz') if filepath.suffix != '.gz' else filepath
        regular_path = filepath.with_suffix('.pkl') if filepath.suffix != '.pkl' else filepath
        
        try:
            if compressed_path.exists():
                with gzip.open(compressed_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded compressed embeddings from {compressed_path}")
                return embeddings_dict
            elif regular_path.exists():
                with open(regular_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded embeddings from {regular_path}")
                return embeddings_dict
            else:
                raise FileNotFoundError(f"No embeddings file found at {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    file_path = '../../../datasets/icd_embeddings/icd_embeddings_dictionary'
    loaded_embeddings = load_pickle(file_path)
    print(f"Loaded {len(loaded_embeddings)} embeddings with pickle")
    return loaded_embeddings

def get_icd_query_embeddings(load_embeddings_flag : bool, model_path : str, batch_size : int, icd_text_descriptions : list[str],  all_queries : list[str], icd_embeddings_path : str, query_embedding_path : str, method : str):


    if load_embeddings_flag == True:
        
        print(f'Loading ICD Description Embeddings')
        icd_sentence_embedding_dictionary = load_embeddings(file_path = icd_descriptons_embedding_path)

        print(f'Loading Query Embeddings')
        query_embedding_dictionary = load_embeddings(file_path = query_embedding_path)
        

    else:

        print(f'Loading Embedding Generator Object')
        embedding_generator_object = Distributed_Sentence_Embeddings(model_path = model_path, batch_size = batch_size)

        print(f'Generating ICD Description Embeddings')
        icd_sentence_embedding_dictionary = embedding_generator_object.generate_sentence_embedding_pair(text_descriptions = icd_text_descriptions)
        print(f'Saving ICD Description Embeddings')
        save_embeddings(embeddings_dict = icd_sentence_embedding_dictionary, filepath = icd_embeddings_path, method = method)

        print(f'Generating Query Embeddings')
        query_embedding_dictionary = embedding_generator_object.generate_sentence_embedding_pair(text_descriptions = all_queries)
        print(f'Saving Query Embeddings')
        save_embeddings(embeddings_dict = query_embedding_dictionary, filepath = query_embedding_path, method = method)
        
    
    return query_embedding_dictionary, icd_sentence_embedding_dictionary

class OptimizedTripletMiner:
    """
    Optimized triplet mining using vectorized operations and FAISS for similarity search
    """
    
    def __init__(self, embedding_dim: int = 1024, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.faiss_index = None
        self.embedding_matrix = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        
    def build_faiss_index(self, embeddings_dict: Dict[str, np.ndarray]):
        """
        Build FAISS index for fast similarity search
        """
        logger.info("Building FAISS index...")
        
        # Convert embeddings dict to matrix
        ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[id_] for id_ in ids]).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.embedding_matrix = embeddings
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        self.idx_to_id = {idx: id_ for idx, id_ in enumerate(ids)}
        
        # Build FAISS index
        if self.use_gpu:
            # GPU version
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for normalized vectors = cosine similarity
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            # CPU version
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built with {len(embeddings)} embeddings")
    
    def get_vectorized_negatives_fast(self, query_embedding: np.ndarray, 
                                    positives: List[str], 
                                    search_pool: List[str], 
                                    num_to_select: int,
                                    similarity_threshold_percentile: float = 0.7) -> List[str]:
        """
        Fast vectorized approach for hard negative mining using FAISS
        """
        # Get indices for search pool
        search_indices = [self.id_to_idx[id_] for id_ in search_pool if id_ in self.id_to_idx]
        
        if len(search_indices) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.copy().astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search for most similar items in the search pool
        k = min(len(search_indices), num_to_select * 10)  # Get more candidates than needed
        similarities, indices = self.faiss_index.search(query_embedding, len(self.embedding_matrix))
        
        # Filter to only include items in search pool
        search_indices_set = set(search_indices)
        filtered_results = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in search_indices_set:
                filtered_results.append((sim, self.idx_to_id[idx]))
            if len(filtered_results) >= k:
                break
        
        if len(filtered_results) == 0:
            return []
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(filtered_results, columns=['similarity', 'id'])
        
        # Calculate threshold based on percentile
        threshold = np.percentile(df['similarity'], similarity_threshold_percentile * 100)
        
        # Select hard negatives (high similarity but not positive)
        hard_negatives = df[df['similarity'] >= threshold]['id'].tolist()
        
        # Select remaining from lower similarity ranges
        easy_negatives = df[df['similarity'] < threshold]['id'].tolist()
        
        # Balance selection
        num_hard = min(len(hard_negatives), num_to_select // 2)
        num_easy = min(len(easy_negatives), num_to_select - num_hard)
        
        selected_negatives = []
        if num_hard > 0:
            selected_negatives.extend(random.sample(hard_negatives, num_hard))
        if num_easy > 0:
            selected_negatives.extend(random.sample(easy_negatives, num_easy))
        
        return selected_negatives

    def get_batch_negatives(self, queries_data: List[Dict], 
                          search_pool: List[str],
                          query_embeddings: Dict[str, np.ndarray],
                          num_to_select: int = 50) -> List[Dict]:
        """
        Process multiple queries in batch for better efficiency
        """
        results = []
        
        for query_data in tqdm(queries_data, desc="Processing queries"):
            query = query_data['query']
            positives = query_data['positives']
            
            if query not in query_embeddings:
                logger.warning(f"Query embedding not found for: {query}")
                continue
            
            query_embedding = query_embeddings[query]
            
            # Get candidate negatives (excluding positives)
            candidate_negatives = [id_ for id_ in search_pool if id_ not in positives]
            
            if len(candidate_negatives) < num_to_select:
                logger.warning(f"Not enough candidates for query: {query}")
                hard_negatives = candidate_negatives
            else:
                hard_negatives = self.get_vectorized_negatives_fast(
                    query_embedding, positives, candidate_negatives, num_to_select
                )
            
            results.append({
                'query': query,
                'positives': positives,
                'negatives': hard_negatives
            })
        
        return results

def construct_triplet_dataset_optimized(filtered_data_list: List[Dict], 
                                      icd_reference_lookup: Dict,
                                      text_sentence_embedding_dictionary: Dict[str, np.ndarray],
                                      query_embeddings: Dict[str, np.ndarray],
                                      top_k: int = 5,
                                      num_to_select: int = 50,
                                      use_gpu: bool = True) -> List[Dict]:
    """
    Optimized triplet dataset construction
    """
    # Initialize the miner
    miner = OptimizedTripletMiner(use_gpu=use_gpu)
    
    # Build FAISS index once
    miner.build_faiss_index(text_sentence_embedding_dictionary)
    
    # Prepare search pool (all ICD descriptions)
    search_pool = list(icd_reference_lookup.values())
    
    hard_negative_triplets = []
    
    # Process data in chunks by specialty
    for i in tqdm(range(len(filtered_data_list)), desc="Processing specialties"):
        specialty_data = filtered_data_list[i]
        specialty = list(specialty_data.keys())[0]
        
        logger.info(f'Processing Specialty: {specialty}')
        
        query_positives_dataset = specialty_data.get(specialty)
        
        # Prepare queries data for batch processing
        queries_data = []
        for query, positives in tqdm(query_positives_dataset.items(), desc=f"Preparing {specialty}"):
            positives = [description for description in positives if description != '']
            positives = positives[:top_k]
            
            if len(positives) > 0:
                queries_data.append({
                    'query': query,
                    'positives': positives
                })
        
        # Process all queries for this specialty in batch
        if queries_data:
            batch_results = miner.get_batch_negatives(
                queries_data, search_pool, query_embeddings, num_to_select
            )
            
            # Format results
            for result in batch_results:
                hard_negative_triplets.append({
                    "specialty": specialty,
                    "anchor": result['query'],
                    "positives": result['positives'],
                    "negatives": result['negatives']
                })
    
    return hard_negative_triplets

# Alternative approach using pure NumPy vectorization (if FAISS is not available)
def get_negatives_vectorized_numpy(query_embedding: np.ndarray,
                                 positives: List[str],
                                 search_pool: List[str],
                                 text_embeddings: Dict[str, np.ndarray],
                                 num_to_select: int) -> List[str]:
    """
    Vectorized negative mining using pure NumPy (fallback if FAISS unavailable)
    """
    # Filter search pool to exclude positives
    candidate_negatives = [id_ for id_ in search_pool if id_ not in positives and id_ in text_embeddings]
    
    if len(candidate_negatives) < num_to_select:
        return candidate_negatives
    
    # Stack all candidate embeddings
    candidate_embeddings = np.stack([text_embeddings[id_] for id_ in candidate_negatives])
    
    # Compute cosine similarity in batch
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    similarities = np.dot(candidate_norms, query_norm)
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'id': candidate_negatives,
        'similarity': similarities
    })
    
    # Sort by similarity (descending)
    df = df.sort_values('similarity', ascending=False)
    
    # Apply percentile-based selection
    similarity_70th = np.percentile(similarities, 70)
    similarity_90th = np.percentile(similarities, 90)
    
    # Hard negatives: between 70th-90th percentile
    hard_negatives = df[(df['similarity'] >= similarity_70th) & 
                       (df['similarity'] < similarity_90th)]['id'].tolist()
    
    # Easy negatives: below 70th percentile
    easy_negatives = df[df['similarity'] < similarity_70th]['id'].tolist()
    
    # Balance selection
    num_hard = min(len(hard_negatives), num_to_select // 2)
    num_easy = min(len(easy_negatives), num_to_select - num_hard)
    
    selected_negatives = []
    if num_hard > 0:
        selected_negatives.extend(np.random.choice(hard_negatives, num_hard, replace=False))
    if num_easy > 0:
        selected_negatives.extend(np.random.choice(easy_negatives, num_easy, replace=False))
    
    return selected_negatives


if __name__ == '__main__':

    file_type = '.json'
    directory_path = "../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/filtered_specialty_query_code_results/"

    print(f'1 : Loding Filtered Annotated Dataset and Query List')
    
    annotations_object = Annotated_DataLoader(file_type = file_type, directory_path = directory_path)
    filtered_data_list, all_queries = annotations_object.load_annotated_files_gpt_41()
    print('Filtered Annotated Dataset and Query List Loaded Successfully')

    print('2 : Loading ICD Reference Lookup')
    icd_reference_file = '../../../../shekhar_tanwar/ICD-ICD-Triplet/dataset/icd10.csv'
    icd_object = ICD_DataLoader(icd_reference_file = icd_reference_file)
    icd_reference_lookup = icd_object.load_icd_files()
    icd_text_descriptions = list(icd_reference_lookup.values())
    print('ICD Reference Lookup Loaded Successfully')

    print('3 : Loading Query and ICD Description Embeddings')
    model_path = '../../../../shekhar_tanwar/ICD-ICD-Triplet/model/NovaSearch_stella_en_1.5B_v5/'
    batch_size = 8

    load_embeddings_flag = False
    method = 'pickle'
    icd_descriptons_embedding_path = '../../../datasets/icd_embeddings/icd_embeddings_dictionary_v2_08082025.pkl.gz'
    query_embedding_path = '../../../datasets/query_embeddings/set3/query_embeddings_dictionary_v3.pkl.gz'
    
    query_embedding_dictionary , icd_sentence_embedding_dictionary = get_icd_query_embeddings(load_embeddings_flag = load_embeddings_flag, 
                                                                                         model_path = model_path, 
                                                                                         batch_size = batch_size, 
                                                                                         icd_text_descriptions = icd_text_descriptions,  
                                                                                         all_queries = all_queries, 
                                                                                         icd_embeddings_path = icd_descriptons_embedding_path , 
                                                                                         query_embedding_path = query_embedding_path, 
                                                                                         method = method)

    print('Query and ICD Description Embeddings Loaded Successfully')
    #Build optimized triplet dataset
    
    print('4 : Build Optimized Triplet Dataset')
    triplet_dataset = construct_triplet_dataset_optimized(
        filtered_data_list=filtered_data_list,
        icd_reference_lookup=icd_reference_lookup,
        text_sentence_embedding_dictionary=icd_sentence_embedding_dictionary,
        query_embeddings=query_embedding_dictionary,
        top_k=10,
        num_to_select=50,
        use_gpu=False  # Set to False if no GPU available
    )
    print('Optimized Triplet Dataset Built Successfully')

    print('5 : Saving Triplets')
    path = '../../../datasets/triplets_v50_250_queries_10positives_50hn/icd_icd_hard_negative_triplets_p10_hn50_08112025.jsonl'
    with open(path, 'w') as outfile:
        for entry in  triplet_dataset:
            json.dump(entry, outfile)
            outfile.write('\n')
    print('Triplets Saved!')
    
    print(f"Generated {len(triplet_dataset)} triplets")

    

    

        
        