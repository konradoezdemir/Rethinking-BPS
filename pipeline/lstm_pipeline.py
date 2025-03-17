# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:20:48 2025
@author: Keyvan Amiri Elyasi
A python script to execute pre-processing, feature extraction, training, and 
evaluation a process transformer model.
"""
import os
import time
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from pre_process.lstm_feat_manager import FeaturesMannager as feat
from pre_process import lstm_embedding_training as em
from pre_process import lstm_samples_creator as sc
from pre_process.lstm_utils import (append_csv_start_end, create_index, 
                                    safe_to_datetime, load_embedded,
                                    align_column_names)
from models import single_task_lstm as STL



class LSTMExperiment():
    def __init__ (self, args=None, cfg=None, run=None, seed=None, logger=None): 
        self.args = args
        self.cfg = cfg
        self.run = run
        self.logger = logger
        self.args.val_ratio = cfg['val_train_ratio']
        # define preprocessing folder
        self.args.processed_path = os.path.join(
            self.args.root_path, 'data', self.args.dataset, 'processed')
        if not os.path.exists(self.args.processed_path):
            os.makedirs(self.args.processed_path)        
        # define model's path
        if args.task == 'rem_time':
            # we rely on models for next activity, timestamp, and rol prediction
            act_name = args.dataset+'#next_act#'+args.sim+'#'+args.model+'#run_'+str(run)+'#seed_'+str(seed)+'#checkpoint.pt'
            time_name = args.dataset+'#next_time#'+args.sim+'#'+args.model+'#run_'+str(run)+'#seed_'+str(seed)+'#checkpoint.pt'
            role_name = args.dataset+'#next_role#'+args.sim+'#'+args.model+'#run_'+str(run)+'#seed_'+str(seed)+'#checkpoint.pt'
            self.args.act_model_path = os.path.join(args.result_path, act_name)
            self.args.time_model_path = os.path.join(args.result_path, time_name)
            self.args.role_model_path = os.path.join(args.result_path, role_name)
            if all(os.path.exists(path) for path in [self.args.act_model_path,
                                                     self.args.time_model_path,
                                                     self.args.role_model_path]):
                self.logger.info('All required models are already trained.')
            else:
                raise ValueError(f'Prediction for {args.task} requires trained models.')                
        else:
            model_name = args.dataset+'#'+args.task+'#'+args.sim+'#'+args.model+'#run_'+str(run)+'#seed_'+str(seed)+'#checkpoint.pt'
            self.args.model_path = os.path.join(args.result_path, model_name)           
        # define path for inference results (a csv file for predictions)
        result_name = args.dataset+'#'+args.task+'#'+args.sim+'#'+args.model+'#run_'+str(run)+'#seed_'+str(seed)+'#inference.csv'
        self.args.inference_path = os.path.join(args.result_path, result_name)
        # necessary initializations
        # Activities and roles indexes
        self.ac_index = dict()
        self.index_ac = dict()
        self.rl_index = dict()
        self.index_rl = dict()
        # Training examples
        self.examples = dict()
        # Embedded dimensions
        self.ac_weights = list()
        self.rl_weights = list()
        # vectorized training, validation and test data
        self.train_vec = dict()
        self.valid_vec = dict()
        self.test_vec =  dict()
        # set dimensions for activity embedding, role embedding, and time to zero
        self.act_dim = 0
        self.role_dim = 0
        # name of the embedding matrices
        self.ac_emb_name = self.args.sim+'_'+self.args.model+'_activity_embedded_matrix_run_'+str(self.run)+'_seed_'+str(seed)+'.emb'
        self.rl_emb_name = self.args.sim+'_'+self.args.model+'_roles_embedded_matrix_run_'+str(self.run)+'_seed_'+str(seed)+'.emb'
        # path for activity and role embedding
        self.ac_emb_path = os.path.join(self.args.processed_path, self.ac_emb_name)
        self.rl_emb_path = os.path.join(self.args.processed_path, self.rl_emb_name)        

    def to_cuda(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return x
    
    def execute_pipeline(self):
        start = time.time()
        self.set_parameters()
        # create the event log
        self.log = self.load_log() 
        # TODO: separate execution pipeline for deep simulator to handle roles
        # Add roles to event log (role discovery)
        self.log = feat.add_resources(self.log, self.parameters['rp_sim'])
        # indexes creation (for activities and roles)
        self.indexing()
        # add relative times (duration and waiting times)
        inp = feat(self.parameters)
        inp.register_scaler(self.parameters['model_type'], self.parameters['scaler'])         
        self.log, self.parameters['scale_args'] = inp.calculate(self.log, []) 
        self.logger.info(f"Scale factors used for temporal features/targets: {self.parameters['scale_args']}")
        # create training, validation and test logs
        self.log_train = self.log[self.log['caseid'].isin(self.train_case_ids)]
        self.log_valid = self.log[self.log['caseid'].isin(self.val_case_ids)]
        self.log_test = self.log[self.log['caseid'].isin(self.test_case_ids)]    
        self.logger.info(f'train split size: {len(self.log_train)}')
        self.logger.info(f'validationn split size: {len(self.log_valid)}')
        self.logger.info(f'test split size: {len(self.log_test)}')        
        # convert training, validation and test logs to feature vectors:
        #self.logger.info(f"ac_index contents: {self.ac_index}")
        #self.logger.info(f"Number of categories: {len(self.ac_index)}")
        vectorizer = sc.SequencesCreator(self.parameters['one_timestamp'],
                                         self.ac_index, self.rl_index, self.logger)
        vectorizer.register_vectorizer(self.parameters['model_type'],
                                       self.parameters['vectorizer'])
        self.train_vec = vectorizer.vectorize(
            self.parameters['model_type'], self.log_train, self.parameters, [])
        self.valid_vec = vectorizer.vectorize(
            self.parameters['model_type'], self.log_valid, self.parameters, [])
        self.test_vec = vectorizer.vectorize(
            self.parameters['model_type'], self.log_test, self.parameters, [])
        (self.case_id_lst, self.prefix_length_lst) = vectorizer.get_inference_lists()
        # Load/train embedded matrix   
        if os.path.exists(self.ac_emb_path):
            self.logger.info('load embedded matrix')
            self.ac_weights = load_embedded(self.index_ac, self.ac_emb_path)
            self.rl_weights = load_embedded(self.index_rl, self.rl_emb_path)
        else:
            self.logger.info('train embeddings')
            em.training_model(
                self.parameters, self.log, self.ac_index, self.index_ac, 
                self.rl_index, self.index_rl, self.ac_emb_path, self.rl_emb_path)
            self.ac_weights = load_embedded(self.index_ac, self.ac_emb_path)
            self.rl_weights = load_embedded(self.index_rl, self.rl_emb_path)
        # get size of activity embedding, role embedding
        self.act_dim = self.ac_weights.shape[-1]
        self.role_dim = self.rl_weights.shape[-1]
        # set number of time features:
        self.time_features = 1 if self.parameters['one_timestamp'] else 2
        self.logger.info(f'GPU available: {torch.cuda.is_available()}')
        # Convert data to DataLoader for batch training
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.convert_data_to_dataloader()
        self.logger.info('defined dataloaders.')
        end = time.time()
        self.logger.info(f'Total processing time (s): {(end - start)}.')
        print('Data pre-processing is done.')        
        if self.args.task != 'rem_time':
            print(f'Train lstm model for {self.args.task}.')
            self.logger.info(f'Train lstm model for {self.args.task}.')
            start = time.time()
            model = STL.MyModel(
                ac_weights=self.ac_weights, rl_weights=self.rl_weights,
                lstm_size=self.parameters['l_size'], act_dim=self.act_dim,
                role_dim=self.role_dim, time_dim =self.time_features,
                task_name=self.args.task )
            model = self.to_cuda(model)
            # convert all floating-point parameters to torch.float64
            model.double()
            print(model)
            # define optimizer
            optimizer = self.set_optimizer(
                model, self.parameters['optim'], self.parameters['base_lr'], 
                self.parameters['eps'], self.parameters['weight_decay'])
            STL.train_model(
                model=model, dataloader=self.train_dataloader, 
                test_dataloader=self.test_dataloader,
                val_dataloader=self.val_dataloader,
                num_epochs=self.parameters['epochs'], optimizer=optimizer,
                task_name=self.args.task, 
                prediction_method=self.parameters['prediction_method'],
                early_patience= self.parameters['patience'], 
                min_delta=self.parameters['min_delta'], 
                ac_index=self.ac_index, rl_index=self.rl_index,
                time_dim =self.time_features,
                dataset_name=self.args.dataset, logger=self.logger, 
                checkpoint_path=self.args.model_path)
            end = time.time()
            self.logger.info(f'Total training time (h): {(end - start)/3600}.')
            print('Training is done.')  
        print(f'Evaluate lstm model for {self.args.task}.')
        self.logger.info(f'Evaluate lstm model for {self.args.task}.')
        start = time.time()
        STL.perform_evaluation(
            args=self.args, params=self.parameters, 
            test_dataloader=self.test_dataloader, test_dataset= self.test_dataset,
            act_dim=self.act_dim, role_dim=self.role_dim, time_dim=self.time_features, 
            ac_index=self.ac_index, rl_index=self.rl_index, 
            ac_weights=self.ac_weights, rl_weights=self.rl_weights,
            rem_time_dict=self.rem_time_dict, case_id_lst=self.case_id_lst,
            prefix_length_lst=self.prefix_length_lst, logger=self.logger)   
        end = time.time()
        self.logger.info(f'Total inference time (s): {(end - start)}.')
        print('Evaluation is done.')        
        
    def set_parameters(self):
        parameters = dict()
        parameters['model_type'] = 'STL'
        parameters['one_timestamp'] = self.cfg['one_timestamp']        
        column_names = self.cfg['column_names']
        timestamp_format = self.cfg['timestamp_format']
        parameters['read_options'] = {'timeformat': timestamp_format,
                                      'column_names': column_names,
                                      'one_timestamp': parameters['one_timestamp']}        
        parameters['rp_sim'] = self.cfg['rp_sim']
        parameters['n_size'] = self.cfg['n_gram_size']
        parameters['norm_method'] = self.cfg['norm_method']
        parameters['scaler'] = self.cfg['scaler']
        parameters['vectorizer'] = self.cfg['vectorizer']
        parameters['l_size'] = self.cfg['layer_size']
        parameters['lstm_act'] = self.cfg['lstm_act']
        parameters['dense_act'] = self.cfg['dense_act']
        parameters['batch_size'] = self.cfg['train']['batch_size']
        parameters['epochs'] = self.cfg['train']['max_epochs']
        parameters['optim'] = self.cfg['train']['optimizer'] 
        parameters['base_lr'] = self.cfg['train']['base_lr']
        parameters['eps'] = float(self.cfg['train']['eps'])
        parameters['weight_decay'] = self.cfg['train']['weight_decay']
        parameters['patience'] = self.cfg['train']['early_stop.patience']
        parameters['min_delta'] = self.cfg['train']['early_stop.min_delta']
        parameters['eval_batch_size'] = self.cfg['evaluation']['batch_size']
        parameters['prediction_method'] = self.cfg['evaluation']['prediction_method']
        # initialize sacling argument and maximum case length parameters
        parameters['scale_args'] = dict()
        parameters['max_length'] = 0
        self.parameters = parameters       


    def load_log(self):
        params = self.parameters['read_options'] 
        column_names = params['column_names']
        timeformat = params['timeformat']        
        # necessary initialization
        data_list = list()
        # load train-validation dataframe
        df_train_val = pd.read_csv(self.args.train_path)
        # load test dataframe
        df_test = pd.read_csv(self.args.test_path)
        # change column names if necessary
        df_train_val = align_column_names(df_train_val)
        df_test = align_column_names(df_test)
        # set string type for case id
        df_train_val['case_id'] = df_train_val['case_id'].astype(str)
        df_test['case_id'] = df_test['case_id'].astype(str)
        # mark train, val, test examples for later split
        df_sorted = df_train_val.sort_values(by='start_timestamp')
        train_val_case_ids = df_sorted['case_id'].drop_duplicates().tolist()
        train_idx = int(len(train_val_case_ids) * (1-self.args.val_ratio))
        self.train_case_ids = train_val_case_ids[:train_idx]
        self.val_case_ids = train_val_case_ids[train_idx:]
        self.test_case_ids = df_test['case_id'].drop_duplicates().tolist()
        # get common columns and concat train-val with test data
        common_cols = df_train_val.columns.intersection(df_test.columns)  
        df_train_val = df_train_val[common_cols]
        df_test = df_test[common_cols] 
        if df_train_val.columns.duplicated().any():
            df_train_val = df_train_val.loc[:, ~df_train_val.columns.duplicated()]
        if df_test.columns.duplicated().any():
            df_test = df_test.loc[:, ~df_test.columns.duplicated()]
        log = pd.concat([df_train_val, df_test], ignore_index=True)
            
        # get necessary data for remaining time prediction
        self.rem_time_dict = {}
        if self.args.task == 'rem_time':
            # max case length in train-val for sequential remaining time prediction
            self.parameters['max_length'] = df_train_val.groupby('case_id').size().max()
            # get ground truth values for remaining time
            df_test['start_timestamp'] = df_test['start_timestamp'].apply(safe_to_datetime)
            df_test['end_timestamp'] = df_test['end_timestamp'].apply(safe_to_datetime)
            df_test['start_timestamp'] = df_test['start_timestamp'].dt.strftime(timeformat)
            df_test['end_timestamp'] = df_test['end_timestamp'].dt.strftime(timeformat)
            df_test['start_timestamp'] = pd.to_datetime(df_test['start_timestamp'])
            df_test['end_timestamp'] = pd.to_datetime(df_test['end_timestamp'])
            sort_key = 'end_timestamp' if self.parameters['one_timestamp'] else 'start_timestamp'
            # Group by case_id
            for case_id, group in df_test.groupby('case_id'):
                group = group.sort_values(sort_key)
                # Get last row's end_timestamp
                last_end = group['end_timestamp'].iloc[-1]  
                for i, (_, row) in enumerate(group.iterrows(), start=1):
                    # Compute time difference in seconds
                    rem_time = (last_end - row['end_timestamp']).total_seconds()
                    rem_time = max(0, rem_time)
                    rem_time = rem_time/3600/24
                    self.rem_time_dict[(case_id, i)] = rem_time
        # rename columns based on configuration file
        log = log.rename(columns=column_names)        
        #log = log[~log.task.isin(['Start', 'End'])]
        # set date time columns
        log['start_timestamp'] = log['start_timestamp'].apply(safe_to_datetime)
        log['end_timestamp'] = log['end_timestamp'].apply(safe_to_datetime)
        log['start_timestamp'] = log['start_timestamp'].dt.strftime(timeformat)
        log['end_timestamp'] = log['end_timestamp'].dt.strftime(timeformat)
        log['start_timestamp'] = pd.to_datetime(log['start_timestamp'])
        log['end_timestamp'] = pd.to_datetime(log['end_timestamp'])
        if (self.args.dataset == 'Confidential_1000' or 
            self.args.dataset == 'Confidential_2000'):
            log['task'] = log['task'].replace('end', 'end#')
            log['task'] = log['task'].replace('start', 'start#') 
        if self.args.dataset == 'ConsultaDataMining':
            log['task'] = log['task'].replace('Start', 'Start#')
            log['task'] = log['task'].replace('End', 'End#')
        data_list = log.to_dict('records')
        data_list = append_csv_start_end(data_list)   
        log_df = pd.DataFrame(data_list)
        if 'role' in log_df.columns:
            log_df = log_df.drop(columns=['role'])
        if 'Unnamed: 0' in log_df.columns:
            log_df = log_df.drop(columns=['Unnamed: 0'])    
        return log_df
        
    def indexing(self):
        # Activities index creation
        self.ac_index = create_index(self.log, 'task')
        self.ac_index['start'] = 0
        self.ac_index['end'] = len(self.ac_index)
        self.index_ac = {v: k for k, v in self.ac_index.items()}
        # Roles index creation
        self.rl_index = create_index(self.log, 'role')
        self.rl_index['start'] = 0
        self.rl_index['end'] = len(self.rl_index)
        self.index_rl = {v: k for k, v in self.rl_index.items()}
        # Add index to the event log
        ac_idx = lambda x: self.ac_index[x['task']]
        self.log['ac_index'] = self.log.apply(ac_idx, axis=1)
        rl_idx = lambda x: self.rl_index[x['role']]
        self.log['rl_index'] = self.log.apply(rl_idx, axis=1)
        
    def convert_data_to_dataloader(self):        
        ac_input = self.to_cuda(torch.tensor(self.train_vec['prefixes']['activities']))
        rl_input = self.to_cuda(torch.tensor(self.train_vec['prefixes']['roles']))
        t_input = self.to_cuda(torch.tensor(self.train_vec['prefixes']['times']))
        act_output = self.to_cuda(torch.tensor(self.train_vec['next_evt']['activities']))
        role_output = self.to_cuda(torch.tensor(self.train_vec['next_evt']['roles']))
        time_output = self.to_cuda(torch.tensor(self.train_vec['next_evt']['times']))
        self.logger.info('Important shapes:')
        self.logger.info(f'ac_input:{ac_input.shape}')
        self.logger.info(f'rl_input:{rl_input.shape}')
        self.logger.info(f't_input:{t_input.shape}')
        self.logger.info(f'act_output:{act_output.shape}')
        self.logger.info(f'role_output:{role_output.shape}')
        self.logger.info(f'time_output:{time_output.shape}')
        train_dataset = TensorDataset(
            ac_input, rl_input, t_input, act_output, role_output, time_output)
        batch_size = self.parameters['batch_size']
        shuffle = True   # Set to True if you want to shuffle the data during training
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle) 
        ac_input = self.to_cuda(torch.tensor(self.valid_vec['prefixes']['activities']))
        rl_input = self.to_cuda(torch.tensor(self.valid_vec['prefixes']['roles']))
        t_input = self.to_cuda(torch.tensor(self.valid_vec['prefixes']['times']))
        act_output = self.to_cuda(torch.tensor(self.valid_vec['next_evt']['activities']))
        role_output = self.to_cuda(torch.tensor(self.valid_vec['next_evt']['roles']))
        time_output = self.to_cuda(torch.tensor(self.valid_vec['next_evt']['times']))     
        valid_dataset = TensorDataset(
            ac_input, rl_input, t_input, act_output, role_output, time_output)
        batch_size = self.parameters['batch_size']
        shuffle = False
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=shuffle)        
        ac_input = self.to_cuda(torch.tensor(self.test_vec['prefixes']['activities']))
        rl_input = self.to_cuda(torch.tensor(self.test_vec['prefixes']['roles']))
        t_input = self.to_cuda(torch.tensor(self.test_vec['prefixes']['times']))
        act_output = self.to_cuda(torch.tensor(self.test_vec['next_evt']['activities']))
        role_output = self.to_cuda(torch.tensor(self.test_vec['next_evt']['roles']))
        time_output = self.to_cuda(torch.tensor(self.test_vec['next_evt']['times']))
        self.test_dataset = TensorDataset(
            ac_input, rl_input, t_input, act_output, role_output, time_output)
        batch_size = self.parameters['eval_batch_size']
        shuffle = False   # Set to True if you want to shuffle the data during test
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle) 
        return train_dataloader, valid_dataloader, test_dataloader
    
    def set_optimizer (self, model, optimizer_type, base_lr, eps, weight_decay):
        if optimizer_type == "NAdam":
            optimizer = optim.NAdam(
                model.parameters(), lr=base_lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == "AdamW":   
            optimizer = optim.AdamW(
                model.parameters(), lr=base_lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == "Adam":   
            optimizer = optim.Adam(
                model.parameters(), lr=base_lr, eps=eps, weight_decay=weight_decay)
        else:
            print('Optimizer type is undefined.')         
        return optimizer
     
        
    def return_args(self):
        return self.args