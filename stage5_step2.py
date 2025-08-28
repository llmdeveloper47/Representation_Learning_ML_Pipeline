# usage
# python3 stage_5_step2.py

import json
import random
import pandas as pd
from datasets import Dataset, load_dataset
from datasets import concatenate_datasets
from sklearn.model_selection import train_test_split

class Flatten_Dataset:
    def __init__(self, dataset_path : str, output_path : str, filter_flag : bool):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.filter_flag = filter_flag
    
    def build_triplet_dataframe(self) -> pd.DataFrame:
        
        triplet_dataset = []
        with open(self.dataset_path) as f:
            for line in f:
                triplet_dataset.append(json.loads(line))
             
        df = pd.DataFrame(triplet_dataset, columns=['specialty', 'anchor', 'positives', 'negatives'])  
        df_flat_pos= df.explode('positives')
        df_flat_neg = df_flat_pos.explode('negatives').reset_index(drop= True)
        df_flattened = df_flat_neg
        
        print(f'Savings Flattened Dataset')
        df_flattened.to_csv(self.output_path)
        
        return df_flattened


    def get_dataset_stats(self, dataset : pd.DataFrame):

        all_specialties = dataset['specialty'].nunique()
        all_queries = dataset['anchor'].nunique()
        all_positives = list(dataset['positives'].unique())
        all_negatives = list(dataset['negatives'].unique())

        all_icd_descriptions = len(list(set(all_positives + all_negatives)))

        print(f'Total Records : {dataset.shape}')
        print(f'Total Specialties : {all_specialties}')
        print(f'Total Queries : {all_queries}')
        print(f'Total ICD Codes : {all_icd_descriptions}')

    def get_train_train_split(self) -> tuple((dict, dict, dict)):
        
        test_size_selected = 0
        if self.filter_flag:
            print('Generating Train Eval Test Dataset with Filtration by anchor sample size')
    
            df = pd.read_csv(self.output_path)[['specialty','anchor','positives','negatives']]
            df.columns = ['specialty','anchor','positives','negatives']
            print(f'Total Dataset Size : {df.shape}')
            all_specialties = df['specialty'].unique().tolist()
    
            print('Current Distribution By Specialty')
            specialty_anchor_distribution = df.groupby(['specialty'])['anchor'].nunique().reset_index()
            print(specialty_anchor_distribution)
    
            # sampling anchors by each specialty
            data_list = []
            for specialty in all_specialties:
                random.seed = 42
                data_specialty = df[df['specialty'] == specialty]
    
                unique_anchors = list(data_specialty['anchor'].unique())
    
                num_samples = 250  # Number of random samples to select
                selected_anchors = random.sample(unique_anchors, num_samples)
                data_specialty = data_specialty[data_specialty['anchor'].isin(selected_anchors)]
                assert data_specialty['anchor'].nunique() == num_samples
                data_list.append(data_specialty)
    
    
            df = pd.concat(data_list)
            print('Uniformed Distribution By Specialty')
            specialty_anchor_distribution = df.groupby(['specialty'])['anchor'].nunique().reset_index()
            print(specialty_anchor_distribution)
    
            print('Splitting Dataset By Specialty')
            data_specialty_dict = {}
            for specialty in all_specialties:
                data_specialty = df[df['specialty'] == specialty][['specialty','anchor','positives','negatives']].drop_duplicates()
                data_specialty_dict[specialty] = data_specialty
    
            train_data_dict = {}
            eval_data_dict = {}
            test_data_dict = {}
            print('Generating Train Eval Test Split By Specialty')
            for specialty in all_specialties:
    
                data = data_specialty_dict[specialty]
    
                all_anchors = data['anchor'].unique().tolist()
                train_anchors, temp_anchors = train_test_split(all_anchors, test_size = 0.005, random_state = 42)
                val_anchors, test_anchors = train_test_split(temp_anchors, test_size = 0.01, random_state = 42)
    
                train_data = data[data['anchor'].isin(train_anchors)]
                eval_data = data[data['anchor'].isin(val_anchors)]
                test_data = data[data['anchor'].isin(test_anchors)]
    
                train_data_dict[specialty] = train_data
                eval_data_dict[specialty] = eval_data
                test_data_dict[specialty] = test_data
    
            return train_data_dict, eval_data_dict, test_data_dict
        
        else:
            
            print('Generating Train Eval Test Dataset without Filtration')
            
            df = pd.read_csv(self.output_path)[['specialty','anchor','positives','negatives']]
            df.columns = ['specialty','anchor','positives','negatives']
            print(f'Total Dataset Size : {df.shape}')
            all_specialties = df['specialty'].unique().tolist()
    
            print('Current Distribution By Specialty')
            specialty_anchor_distribution = df.groupby(['specialty'])['anchor'].nunique().reset_index()
            print(specialty_anchor_distribution)
    
            print('Splitting Dataset By Specialty')
            data_specialty_dict = {}
            for specialty in all_specialties:
                data_specialty = df[df['specialty'] == specialty][['specialty','anchor','positives','negatives']].drop_duplicates()
                data_specialty_dict[specialty] = data_specialty
    
            train_data_dict = {}
            eval_data_dict = {}
            test_data_dict = {}
            print('Generating Train Eval Test Split By Specialty')
            for specialty in all_specialties:
    
                data = data_specialty_dict[specialty]
    
                all_anchors = data['anchor'].unique().tolist()
    
                n_samples = len(all_anchors)
                
                if n_samples <= 2:
                    
                    train_anchors = all_anchors
                    val_anchors = []
                    test_anchors = []
                else:
                    
                    test_size = min(0.005, max(1/n_samples if n_samples > 0 else 0.01, 0.01))
                    #print(f'Test Size Selected : {test_size}')
                    combined_test_size = min(test_size * 2, 0.5)
                    
                    train_anchors, temp_anchors = train_test_split(all_anchors, test_size = combined_test_size , random_state = 42)
                
    
                    if len(temp_anchors) == 1:
                        val_anchors = temp_anchors
                        test_anchors = []
                    else:
                        val_anchors, test_anchors = train_test_split(temp_anchors, test_size = 0.5, random_state = 42)
    
                train_data = data[data['anchor'].isin(train_anchors)]
                eval_data = data[data['anchor'].isin(val_anchors)]if val_anchors else pd.DataFrame(columns = data.columns)
                test_data = data[data['anchor'].isin(test_anchors)] if test_anchors else pd.DataFrame(columns = data.columns)
    
                train_data_dict[specialty] = train_data
                eval_data_dict[specialty] = eval_data
                test_data_dict[specialty] = test_data
    
            return train_data_dict, eval_data_dict, test_data_dict

    def get_combined_train_eval_test_datasets(self, train_data_dict : dict, eval_data_dict : dict, test_data_dict : dict, train_output_path : str, eval_output_path : str, test_output_path : str) -> tuple((pd.DataFrame, pd.DataFrame, pd.DataFrame)):
    
    
        all_specialties = list(train_data_dict.keys())
        
        train_data_list = []
        eval_data_list = []
        test_data_list = []
        
        for specialty in all_specialties:
            
            train_data_specialty = train_data_dict[specialty]
            eval_data_specialty = eval_data_dict[specialty]
            test_data_specialty = test_data_dict[specialty]
            
            train_data_list.append(train_data_specialty)
            eval_data_list.append(eval_data_specialty)
            test_data_list.append(test_data_specialty)
            
        
        train_dataset = pd.concat(train_data_list)
        eval_dataset = pd.concat(eval_data_list)
        test_dataset = pd.concat(test_data_list)
        
        train_dataset = train_dataset.drop_duplicates()
        eval_dataset = eval_dataset.drop_duplicates()
        test_dataset = test_dataset.drop_duplicates()
        
        train_dataset = train_dataset.reset_index().iloc[:,1:]
        eval_dataset = eval_dataset.reset_index().iloc[:,1:]
        test_dataset = test_dataset.reset_index().iloc[:,1:]

        train_data = train_data.dropna()
        eval_data = eval_data.dropna()
        test_data = test_data.dropna()
        
        print(f'Total Train Dataset Size : {train_dataset.shape}')
        print(f'Total Eval Dataset Size : {eval_dataset.shape}')
        print(f'Total Test Dataset Size : {test_dataset.shape}')

        train_dataset.to_csv(train_output_path)
        eval_dataset.to_csv(eval_output_path)
        test_dataset.to_csv(test_output_path)
        
        return train_dataset, eval_dataset, test_dataset


    def build_huggingface_dataset(self, TRAIN_DATASET_PATH : str, EVAL_DATASET_PATH : str, TEST_DATASET_PATH : str):
        data_files = {
            'train': TRAIN_DATASET_PATH,
            'eval':  EVAL_DATASET_PATH,
            'test':  TEST_DATASET_PATH,
        }
        # Streaming reads rows lazily from disk
        train_dataset = load_dataset('csv', data_files=data_files, split = 'train') #, streaming=True)
        eval_dataset = load_dataset('csv', data_files=data_files, split = 'eval')
        test_dataset = load_dataset('csv', data_files=data_files, split = 'test')
        
        # Remove only extraneous columns (if any); retain anchor/positives/negatives
        train_dataset = train_dataset.select_columns(['anchor', 'positives', 'negatives'])
        eval_dataset = eval_dataset.select_columns(['anchor', 'positives', 'negatives'])
        test_dataset = test_dataset.select_columns(['anchor', 'positives', 'negatives'])
    
        model_path = '../../../../shekhar_tanwar/ICD-ICD-Triplet/model/e5-large-v2-20250331143312-finetuned-icd-v30/'
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
        def add_triplet_length(batch):
            # Concatenate to one long list to tokenize once.
            n = len(batch['anchor'])
    
            anchors = [str(x) if x is not None else "" for x in batch["anchor"]]
            positives = [str(x) if x is not None else "" for x in batch["positives"]]
            negatives = [str(x) if x is not None else "" for x in batch["negatives"]]
            
            all_texts = anchors + positives + negatives
        
            # Fast batch tokenization; ask only for lengths
            out = tokenizer(
                all_texts,
                add_special_tokens=False,
                return_length=True,
                padding=False,
                truncation=False
            )
            lens = np.asarray(out['length'], dtype=np.int32)
        
            # Slice back by role
            anchor_len   = lens[:n]
            positive_len = lens[n:2*n]
            negative_len = lens[2*n:]
        
            # Max length across the triplet
            triplet_len = np.maximum.reduce([anchor_len, positive_len, negative_len])
        
            return {'triplet_length': triplet_len.tolist()}
    
    
        # Choose large batch size; tune based on RAM/CPU cache
        BATCHED_SIZE = 8192  # try 8k, 16k, 32k
        NUM_PROC = max(2, os.cpu_count() - 2)
    
        train_dataset = train_dataset.map(
            add_triplet_length,
            batched=True,
            batch_size=BATCHED_SIZE,
            num_proc=NUM_PROC,
            remove_columns=[],          # don't drop anything
            desc="Computing Train triplet lengths"
        )
    
        eval_dataset = eval_dataset.map(
            add_triplet_length,
            batched=True,
            batch_size=BATCHED_SIZE,
            num_proc=NUM_PROC,
            remove_columns=[],          # don't drop anything
            desc="Computing Eval triplet lengths"
        )
    
        test_dataset = test_dataset.map(
            add_triplet_length,
            batched=True,
            batch_size=BATCHED_SIZE,
            num_proc=NUM_PROC,
            remove_columns=[],          # don't drop anything
            desc="Computing Test triplet lengths"
        )

    
        return train_dataset, eval_dataset, test_dataset

    def get_batches_by_length(data):
        
    
        b0 = data.filter(lambda ex : ex['triplet_length'] <= 64).shuffle(seed = 42)
        b1 = data.filter(lambda ex : 64 < ex['triplet_length'] <= 128).shuffle(seed = 42)
        b2 = data.filter(lambda ex : 128 < ex['triplet_length']).shuffle(seed = 42)
    
        batched_data_by_length = concatenate_datasets([b0, b1, b2])
    
        return batched_data_by_length
    

if __name__ == "__main__":


    triplet_dataset_path = '../../../datasets/triplets_v50_250_queries_10positives_50hn/icd_icd_hard_negative_triplets_p10_hn50_08112025.jsonl'
    flattened_dataset_output_path = '../../../datasets/dataset_training/triplet_dataset_v50_250_queries_10positives_50hn_08112025.csv'
    filter_flag = False
    
    flatten_object = Flatten_Dataset(dataset_path = triplet_dataset_path, output_path = flattened_dataset_output_path, filter_flag = filter_flag)

    print('1 : Generating Flattened Dataset')
    df_flattened = flatten_object.build_triplet_dataframe()
    print('Flattened Dataset Generated')

    print('2 : Generating Dataset Summary Statistics')
    flatten_object.get_dataset_stats(dataset = df_flattened)
    print('Dataset Summary Generated')

    print('3 : Generating Train Test Eval Split')
    
    train_data_dict, eval_data_dict, test_data_dict = flatten_object.get_train_train_split()

    train_output_path = '../../../datasets/dataset_training/triplet_dataset_v50_250_queries_10positives_50hn_train_08112025.csv'
    eval_output_path = '../../../datasets/dataset_training/triplet_dataset_v50_250_queries_10positives_50hn_eval_08112025.csv'
    test_output_path = '../../../datasets/dataset_training/triplet_dataset_v50_250_queries_10positives_50hn_test_08112025.csv'
    
    
    train_dataset_df, eval_dataset_df, test_dataset_df = flatten_object.get_combined_train_eval_test_datasets(train_data_dict = train_data_dict, eval_data_dict = eval_data_dict, test_data_dict = test_data_dict, train_output_path = train_output_path, eval_output_path = eval_output_path, test_output_path = test_output_path)
    print('Train Test Eval Split Generated')         

    train_dataset, eval_dataset, test_dataset = flatten_object.build_huggingface_dataset(TRAIN_DATASET_PATH = train_output_path, EVAL_DATASET_PATH = eval_output_path, TEST_DATASET_PATH = test_output_path)

    
    train_dataset = flatten_object.get_batches_by_length(train_dataset)
    eval_dataset =  flatten_object.get_batches_by_length(eval_dataset)
    test_dataset =  flatten_object.get_batches_by_length(test_dataset)

    output_dir = '../../../datasets/dataset_training/triplet_v50_splits'

    train_dataset.save_to_disk(f'{output_dir}/train')
    eval_dataset.save_to_disk(f'{output_dir}/eval')
    test_dataset.save_to_disk(f'{output_dir}/test')
    

    
    
        
        
                
                        
                
                        
                        
