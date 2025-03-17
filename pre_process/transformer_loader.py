# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 06:55:47 2025
This script is based on the following source code:
    https://github.com/Zaharah/processtransformer
We adjusted some parts to efficiently use it in our experiments.
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing 

class LogsDataLoader:
    def __init__(self, args=None):
        self.args = args

    def load_data(self):
        if (self.args.task not in ['next_act', 'next_time', 'rem_time', 'next_role']):
            raise ValueError("Invalid task.")
        train_df = pd.read_csv(self.args.processed_train_path)
        test_df = pd.read_csv(self.args.processed_test_path)       
        with open(self.args.meta_data_path, "r") as json_file:
            metadata = json.load(json_file)
        if self.args.task != 'next_role':
            x_word_dict = metadata["x_word_dict"]
            y_word_dict = metadata["y_word_dict"]
        else:
            x_word_dict = metadata["x_word_dict_resource"]
            y_word_dict = metadata["y_word_dict_resource"]
        vocab_size = len(x_word_dict) 
        total_classes = len(y_word_dict)
        max_case_length = self.get_max_case_length(train_df["prefix"].values)
        return (train_df, test_df, 
            x_word_dict, y_word_dict, 
            max_case_length, vocab_size, 
            total_classes)

    def prepare_data_next_activity(self, df, x_word_dict, y_word_dict, 
                                   max_case_length, inference=False):        
        x = df["prefix"].values
        y = df["next_act"].values
        if not inference:
            x, y = utils.shuffle(x, y)
        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)
        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)
        return (token_x, token_y)
    
    def prepare_data_next_role(self, df, x_word_dict, y_word_dict, 
                                   max_case_length, inference=False):        
        x = df["prefix"].values
        y = df["next_role"].values
        if not inference:
            x, y = utils.shuffle(x, y)
        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)
        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)
        return (token_x, token_y)
    
    
    def prepare_data_next_time(self, df, x_word_dict,  max_case_length, 
                               time_scaler = None, y_scaler = None, 
                               inference=False):
        x = df["prefix"].values
        time_x = df[
            ['recent_wait_time', 'recent_proc_time', 'latest_wait_time',
             'latest_proc_time','time_passed']].values.astype(np.float32)  
        y = df[['next_wait_time', 'next_proc_time']].values.astype(np.float32) 
        if not inference:
            x, time_x, y = utils.shuffle(x, time_x, y)
        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)       
        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y).astype(np.float32)
        else:
            y = y_scaler.transform(y).astype(np.float32)
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)        
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return (token_x, time_x, y, time_scaler, y_scaler)
    

    def prepare_data_remaining_time(self, df, x_word_dict, max_case_length,
                                    time_scaler = None, y_scaler = None,
                                    inference=False):
        x = df['prefix'].values
        time_x = df[
            ['recent_wait_time', 'recent_proc_time', 'latest_wait_time',
             'latest_proc_time','time_passed']].values.astype(np.float32)          
        y = df['remaining_time'].values.astype(np.float32)
        if not inference:
            x, time_x, y = utils.shuffle(x, time_x, y)
        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(
                time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(
                time_x).astype(np.float32)       
        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(
                y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(
                y.reshape(-1, 1)).astype(np.float32)
        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)        
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)        
        return (token_x, time_x, y, time_scaler, y_scaler)

    def get_max_case_length(self, train_x):
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x)
   
    def return_args(self):
        return self.args
