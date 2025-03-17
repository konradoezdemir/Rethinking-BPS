# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 06:22:39 2025
This script is based on the following source code:
    https://github.com/Zaharah/processtransformer
We adjusted some parts to seamlessly integrate it into our experiments.
"""

import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import  Pool

from pre_process.lstm_utils import align_column_names, safe_to_datetime
from pre_process import lstm_role_discovery as rl


class LogsDataProcessor:
    def __init__(self, args = None, run=None, seed=None, columns=None,
                 logger=None, pool = 1):
        self.args = args
        self.run = run
        self.seed = seed
        self._org_columns = columns
        self.logger = logger
        self._pool = pool
        # path to save metadata
        meta_name = self.args.dataset+'#'+self.args.sim+'#'+self.args.task+'#run'+str(run)+'#metadata.json'
        self.args.meta_data_path = os.path.join(self.args.processed_path, meta_name) 
        
    def process_logs(self):
        train_df = self._load_df(path=self.args.train_path, sort_temporally=True)
        test_df = self._load_df(path=self.args.test_path, sort_temporally=True)        
        # compute meta data 
        if not os.path.exists(self.args.meta_data_path): 
            #train_df_org = self._load_df(path=self.args.train_path_org, sort_temporally=True)         
            #df = pd.concat([train_df_org, test_df], axis=0, ignore_index=True)
            df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            self._extract_logs_metadata(df)
        
        train_df_split = np.array_split(train_df, self._pool)
        test_df_split = np.array_split(test_df, self._pool)
        # addresses for processed event logs as csv files
        train_suffix = self.args.sim+'_'+self.args.model+'_'+self.args.task+'_train_run_'+str(self.run)+'_seed_'+str(self.seed)+'.csv'
        test_suffix = self.args.sim+'_'+self.args.model+'_'+self.args.task+'_test_run_'+str(self.run)+'_seed_'+str(self.seed)+'.csv'
        train_path = os.path.join(self.args.processed_path, train_suffix)
        test_path = os.path.join(self.args.processed_path, test_suffix)
        # save pathes to remove them later
        self.args.temp_files.append(train_path)
        self.args.temp_files.append(test_path)
        if self.args.task == 'next_act':
            with Pool(processes=self._pool) as pool:
                processed_train_df = pd.concat(
                    pool.imap_unordered(self._next_activity_helper_func, train_df_split))
            with Pool(processes=self._pool) as pool:
                processed_test_df = pd.concat(
                    pool.imap_unordered(self._next_activity_helper_func, test_df_split))           
        elif self.args.task == 'next_role':
            with Pool(processes=self._pool) as pool:
                processed_train_df = pd.concat(
                    pool.imap_unordered(self._next_role_helper_func, train_df_split))
            with Pool(processes=self._pool) as pool:
                processed_test_df = pd.concat(
                    pool.imap_unordered(self._next_role_helper_func, test_df_split))     
        elif self.args.task == 'next_time':
            with Pool(processes=self._pool) as pool:
                processed_train_df = pd.concat(
                    pool.imap_unordered(self._next_time_helper_func, train_df_split))
            with Pool(processes=self._pool) as pool:
                processed_test_df = pd.concat(
                    pool.imap_unordered(self._next_time_helper_func, test_df_split))   
        elif self.args.task == 'rem_time':
            with Pool(processes=self._pool) as pool:
                processed_train_df = pd.concat(
                    pool.imap_unordered(self._remaining_time_helper_func, train_df_split))
            with Pool(processes=self._pool) as pool:
                processed_test_df = pd.concat(
                    pool.imap_unordered(self._remaining_time_helper_func, test_df_split))                
        else:
            print(self.args.task)
            raise ValueError("Invalid task.")
        processed_train_df.to_csv(train_path, index = False)
        processed_test_df.to_csv(test_path, index = False)
        self.args.processed_train_path = train_path
        self.args.processed_test_path = test_path
        
    
    def _load_df(self, path=None, sort_temporally=True):
        df = pd.read_csv(path)
        # adjust column names
        df = align_column_names(df)
        # drop duplicated columns
        df = df.loc[:, ~df.columns.duplicated()]
        # remove start and end if they have been added to dataset.
        # df = df[~df.activity_name.isin(['Start', 'End'])]        
        if (self.args.dataset == 'Confidential_1000' or 
            self.args.dataset == 'Confidential_2000'):
            df['activity_name'] = df['activity_name'].replace('end', 'end#')
            df['activity_name'] = df['activity_name'].replace('start', 'start#') 
        if self.args.dataset == 'ConsultaDataMining':
            df['activity_name'] = df['activity_name'].replace('Start', 'Start#')
            df['activity_name'] = df['activity_name'].replace('End', 'End#')           
        df = df[self._org_columns]

        if self.args.task != 'next_role':
            df.drop('resource', axis=1, inplace=True)
            df.columns = ['case:concept:name', 'concept:name',
                          'time:timestamp', 'start_timestamp']
        else:
            df.columns = ['case:concept:name', 'task', 'time:timestamp',
                          'start_timestamp', 'user'] 
            # discover roles and add them to the dataframe
            res_analyzer = rl.ResourcePoolAnalyser(df, sim_threshold=self.args.rp_sim)
            resources = pd.DataFrame.from_records(res_analyzer.resource_table)
            # Add roles information
            resources = resources.rename(columns={'resource': 'user'})
            df = df.merge(resources, on='user', how='left')
            df = df.reset_index(drop=True)
            # get columns in order again
            df = df.rename(columns={'task': 'concept:name'})        
            df = df[['case:concept:name', 'concept:name', 'time:timestamp',
                     'start_timestamp', 'role']]
        df['concept:name'] = df['concept:name'].str.lower()
        df['concept:name'] = df['concept:name'].str.replace(" ", "-") 
        if self.args.task == 'next_role':
            df['role'] = df['role'].str.lower()
            df['role'] = df['role'].str.replace(" ", "-") 
        df['time:timestamp'] = df['time:timestamp'].apply(safe_to_datetime)
        df['time:timestamp'] = df['time:timestamp'].dt.strftime(self.args.time_format)
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])   
        df['start_timestamp'] = df['start_timestamp'].apply(safe_to_datetime)
        df['start_timestamp'] = df['start_timestamp'].dt.strftime(self.args.time_format)
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
        if sort_temporally:
            # TODO: it might be better to sort based on start timestamp
            df.sort_values(by = ['time:timestamp'], inplace = True)
        return df

    def _extract_logs_metadata(self, df):
        
        # extract meta data for activities
        keys = ["[PAD]", "[UNK]"]
        activities = list(df['concept:name'].unique())
        keys.extend(activities)
        val = range(len(keys))
        coded_activity = dict({"x_word_dict":dict(zip(keys, val))})
        code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})
        coded_activity.update(code_activity_normal)
        
        # extract meta data for roles
        if self.args.task == 'next_role':
            resource_keys = ["[PAD]", "[UNK]"]
            resources = list(df['role'].unique())
            resource_keys.extend(resources)
            resource_val = range(len(resource_keys))
            coded_resource = dict({"x_word_dict_resource":dict(zip(resource_keys, resource_val))})
            code_resource_normal = dict({"y_word_dict_resource": dict(zip(resources, range(len(resources))))})
            coded_activity.update(coded_resource)
            coded_activity.update(code_resource_normal)
        
        # save metadata as a json file
        coded_json = json.dumps(coded_activity)   
        with open(self.args.meta_data_path, "w") as metadata_file:
            metadata_file.write(coded_json)
        

    def _next_activity_helper_func(self, df):
        case_id, act_name = 'case:concept:name', 'concept:name'
        processed_df = pd.DataFrame(columns = ['case_id', 'prefix', 'k', 'next_act'])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][act_name].to_list()
            for i in range(len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))        
                next_act = act[i+1]
                processed_df.at[idx, 'case_id']  =  case
                processed_df.at[idx, 'prefix']  =  prefix
                processed_df.at[idx, 'k'] =  i+1
                processed_df.at[idx, 'next_act'] = next_act
                idx = idx + 1
        return processed_df
    
    def _next_role_helper_func(self, df):
        case_id, role_name = 'case:concept:name', 'role'
        processed_df = pd.DataFrame(columns = ['case_id', 'prefix', 'k', 'next_role'])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            role = df[df[case_id] == case][role_name].to_list()
            for i in range(len(role) - 1):
                prefix = np.where(i == 0, role[0], " ".join(role[:i+1]))        
                next_role = role[i+1]
                processed_df.at[idx, 'case_id']  =  case
                processed_df.at[idx, 'prefix']  =  prefix
                processed_df.at[idx, 'k'] =  i+1
                processed_df.at[idx, 'next_role'] = next_role
                idx = idx + 1
        return processed_df



    def _next_time_helper_func(self, df):
        case_id = 'case:concept:name'
        event_name = 'concept:name'
        event_time = 'time:timestamp'
        event_time_start = 'start_timestamp'
        df[event_time] = df[event_time].astype(str)
        df[event_time_start] = df[event_time_start].astype(str)
        processed_df = pd.DataFrame(columns = [
            'case_id', 'prefix', 'k', 'time_passed', 'recent_wait_time',
            'recent_proc_time', 'latest_wait_time', 'latest_proc_time',
            'next_wait_time', 'next_proc_time'])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time2 = df[df[case_id] == case][event_time_start].str[:19].to_list()
            time_passed = 0
            latest_diff_wait = datetime.timedelta()
            recent_diff_wait = datetime.timedelta()
            next_time_wait =  datetime.timedelta()
            latest_diff_proc = datetime.timedelta()
            recent_diff_proc = datetime.timedelta()
            next_time_proc =  datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                latest_diff_proc = datetime.datetime.strptime(time[i], self.args.time_format) - \
                    datetime.datetime.strptime(time2[i], self.args.time_format)   
                if i > 0:
                    latest_diff_wait = datetime.datetime.strptime(time2[i], self.args.time_format) - \
                                        datetime.datetime.strptime(time[i-1], self.args.time_format)  
                    recent_diff_proc = datetime.datetime.strptime(time[i-1], self.args.time_format) - \
                                        datetime.datetime.strptime(time2[i-1], self.args.time_format)
                if i > 1:
                    recent_diff_wait = datetime.datetime.strptime(time2[i-1], self.args.time_format)- \
                                    datetime.datetime.strptime(time[i-2], self.args.time_format)
                #latest_time_proc = latest_diff_proc.days
                latest_time_proc = int(latest_diff_proc.total_seconds())              
                #latest_time_wait = np.where(i == 0, 0, latest_diff_wait.days)
                latest_time_wait = np.where(i == 0, 0, int(latest_diff_wait.total_seconds()))
                #recent_time_proc = np.where(i == 0, 0, recent_diff_proc.days)
                recent_time_proc = np.where(i == 0, 0, int(recent_diff_proc.total_seconds()))
                #recent_time_wait = np.where(i <=1, 0, recent_diff_wait.days)
                recent_time_wait = np.where(i <=1, 0, int(recent_diff_wait.total_seconds()))
                time_passed = time_passed + latest_time_proc + latest_time_wait
                if i+1 < len(time):
                    next_time_wait = datetime.datetime.strptime(time2[i+1], self.args.time_format) - \
                        datetime.datetime.strptime(time[i], self.args.time_format)
                    next_time_proc = datetime.datetime.strptime(time[i+1], self.args.time_format) - \
                        datetime.datetime.strptime(time2[i+1], self.args.time_format)
                    next_time_wait_days = str(int(next_time_wait.total_seconds())) 
                    #next_time_wait_days = str(int(next_time_wait.days))                               
                    next_time_proc_days = str(int(next_time_proc.total_seconds()))
                    #next_time_proc_days = str(int(next_time_proc.days))
                else:
                    next_time_wait_days = str(1) 
                    next_time_proc_days = str(1)
                processed_df.at[idx, 'case_id']  = case
                processed_df.at[idx, 'prefix']  =  prefix
                processed_df.at[idx, 'k'] = i+1
                processed_df.at[idx, 'time_passed'] = time_passed
                processed_df.at[idx, 'recent_wait_time'] = recent_time_wait
                processed_df.at[idx, 'recent_proc_time'] = recent_time_proc
                processed_df.at[idx, 'latest_wait_time'] =  latest_time_wait
                processed_df.at[idx, 'latest_proc_time'] =  latest_time_proc
                processed_df.at[idx, 'next_wait_time'] = next_time_wait_days
                processed_df.at[idx, 'next_proc_time'] = next_time_proc_days
                idx = idx + 1
        processed_df_time = processed_df[
            ['case_id', 'prefix', 'k', 'time_passed', 'recent_wait_time',
             'recent_proc_time', 'latest_wait_time', 'latest_proc_time',
             'next_wait_time', 'next_proc_time']]
        return processed_df_time



    def _remaining_time_helper_func(self, df):
        case_id = 'case:concept:name'
        event_name = 'concept:name'
        event_time = 'time:timestamp'
        event_time_start = 'start_timestamp'
        df[event_time] = df[event_time].astype(str)
        df[event_time_start] = df[event_time_start].astype(str)
        processed_df = pd.DataFrame(columns = [
            'case_id', 'prefix', 'k', 'time_passed', 'recent_wait_time',
            'recent_proc_time', 'latest_wait_time', 'latest_proc_time',
            'remaining_time'])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time2 = df[df[case_id] == case][event_time_start].str[:19].to_list()
            time_passed = 0
            latest_diff_wait = datetime.timedelta()
            recent_diff_wait = datetime.timedelta()
            latest_diff_proc = datetime.timedelta()
            recent_diff_proc = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                latest_diff_proc = datetime.datetime.strptime(time[i], self.args.time_format) - \
                    datetime.datetime.strptime(time2[i], self.args.time_format)  
                if i > 0:
                    latest_diff_wait = datetime.datetime.strptime(time2[i], self.args.time_format) - \
                                        datetime.datetime.strptime(time[i-1], self.args.time_format)  
                    recent_diff_proc = datetime.datetime.strptime(time[i-1], self.args.time_format) - \
                                        datetime.datetime.strptime(time2[i-1], self.args.time_format)
                if i > 1:
                    recent_diff_wait = datetime.datetime.strptime(time2[i-1], self.args.time_format)- \
                                    datetime.datetime.strptime(time[i-2], self.args.time_format)
               
                #latest_time_proc = latest_diff_proc.days
                latest_time_proc = int(latest_diff_proc.total_seconds())                
                #latest_time_wait = np.where(i == 0, 0, latest_diff_wait.days)
                latest_time_wait = np.where(i == 0, 0, int(latest_diff_wait.total_seconds()))
                #recent_time_proc = np.where(i == 0, 0, recent_diff_proc.days)
                recent_time_proc = np.where(i == 0, 0, int(recent_diff_proc.total_seconds()))
                #recent_time_wait = np.where(i <=1, 0, recent_diff_wait.days)
                recent_time_wait = np.where(i <=1, 0, int(recent_diff_wait.total_seconds()))
                time_passed = time_passed + latest_time_proc + latest_time_wait  

                time_stamp = str(np.where(i == 0, time[0], time[i]))
                ttc = datetime.datetime.strptime(time[-1], self.args.time_format) - \
                        datetime.datetime.strptime(time_stamp, self.args.time_format)
                #ttc = str(ttc.days)
                ttc = str(int(ttc.total_seconds()))

                processed_df.at[idx, 'case_id']  = case
                processed_df.at[idx, 'prefix']  =  prefix
                processed_df.at[idx, 'k'] = i+1
                processed_df.at[idx, 'time_passed'] = time_passed
                processed_df.at[idx, 'recent_wait_time'] = recent_time_wait
                processed_df.at[idx, 'recent_proc_time'] = recent_time_proc
                processed_df.at[idx, 'latest_wait_time'] =  latest_time_wait
                processed_df.at[idx, 'latest_proc_time'] =  latest_time_proc
                processed_df.at[idx, 'remaining_time'] = ttc               
                idx = idx + 1
        processed_df_remaining_time = processed_df[['case_id', 'prefix', 'k', 
            'time_passed', 'recent_wait_time', 'recent_proc_time',
            'latest_wait_time', 'latest_proc_time', 'remaining_time']]
        return processed_df_remaining_time
    
    def return_args(self):
        return self.args