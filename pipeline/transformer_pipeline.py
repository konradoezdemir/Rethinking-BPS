# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:10:22 2025
A python script to execute pre-processing, feature extraction, training, and 
evaluation a process transformer model.
"""
import os
# TODO: probably get rid of these two lines
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import tensorflow as tf
from sklearn import metrics 

from pre_process.transformer_processor import LogsDataProcessor
from pre_process.transformer_loader import LogsDataLoader
from models import transformer
from pre_process.lstm_utils import safe_mape


class TransformerExperiment():
    def __init__ (self, args=None, cfg=None, run=None, seed=None, logger=None): 
        self.args = args
        self.args.epochs = cfg['epochs']
        self.args.batch_size = cfg['batch_size']
        self.args.learning_rate = cfg['learning_rate']
        self.args.time_format = cfg['timestamp_format']  
        self.args.rp_sim = cfg['rp_sim']
        self.run = run
        self.seed = seed
        self.logger = logger
        # define preprocessing folder
        self.args.processed_path = os.path.join(
            self.args.root_path, 'data', self.args.dataset, 'processed')
        if not os.path.exists(self.args.processed_path):
            os.makedirs(self.args.processed_path) 
        # define model's path
        model_name = args.dataset+'#'+args.task+'#'+args.sim+'#'+args.model+'#run_'+str(run)+'#seed_'+str(seed)+'.weights.h5'
        self.args.model_path = os.path.join(args.result_path, model_name)
        # define path for inference results (a csv file for predictions)
        result_name = args.dataset+'#'+args.task+'#'+args.sim+'#'+args.model+'#run_'+str(run)+'#seed_'+str(seed)+'#inference.csv'
        self.args.inference_path = os.path.join(args.result_path, result_name)
        self.args.temp_files = []
        
    def execute_pipeline(self):
        # Check if GPU is available and TensorFlow is built with CUDA
        self.logger.info(f'GPU available: {tf.test.is_gpu_available()}')
        self.logger.info(f'TensorFlow is built with CUDA: {tf.test.is_built_with_cuda()}')
        start = time.time()
        data_processor = LogsDataProcessor(
            args=self.args, run=self.run, seed=self.seed, 
            columns = ['case_id', 'activity_name', 'end_timestamp',
                       'start_timestamp', 'resource'],
            logger= self.logger, pool = 1)
        data_processor.process_logs()
        end = time.time()
        self.logger.info(f'Total processing time (s): {(end - start)}.')
        self.args = data_processor.return_args()
        
        start = time.time()
        data_loader = LogsDataLoader(args=self.args)
        (self.train_df, self.test_df, self.x_word_dict, self.y_word_dict,
         self.max_case_length, self.vocab_size, self.num_output
         ) = data_loader.load_data()
        if self.args.task == 'next_act':
            # Prepare training examples for next activity prediction task
            (self.train_token_x, self.train_token_y
             ) = data_loader.prepare_data_next_activity(
                 self.train_df, self.x_word_dict, self.y_word_dict,
                 self.max_case_length)
            # Prepare test examples for next activity prediction task
            (self.test_token_x, self.test_token_y
             ) = data_loader.prepare_data_next_activity(
                 self.test_df, self.x_word_dict, self.y_word_dict,
                 self.max_case_length, inference=True)
        elif self.args.task == 'next_role':
            # Prepare training examples for next role prediction task
            (self.train_token_x, self.train_token_y
             ) = data_loader.prepare_data_next_role(
                 self.train_df, self.x_word_dict, self.y_word_dict,
                 self.max_case_length)
            # Prepare test examples for next role prediction task
            (self.test_token_x, self.test_token_y
             ) = data_loader.prepare_data_next_role(
                 self.test_df, self.x_word_dict, self.y_word_dict,
                 self.max_case_length, inference=True) 
        elif self.args.task == 'next_time':
            # Prepare training examples for next time prediction task
            (self.train_token_x, self.train_time_x, self.train_token_y, 
             self.time_scaler, self.y_scaler
             ) = data_loader.prepare_data_next_time(
                 self.train_df, self.x_word_dict, self.max_case_length)
            # Prepare test examples for next time prediction task
            (self.test_token_x, self.test_time_x, self.test_y, _, _
             ) = data_loader.prepare_data_next_time(
                 self.test_df, self.x_word_dict, self.max_case_length, 
                 self.time_scaler, self.y_scaler, inference=True)                                                                                
        elif self.args.task == 'rem_time':
            # Prepare training examples for remaining time prediction task
            (self.train_token_x, self.train_time_x, self.train_token_y,
             self.time_scaler, self.y_scaler
             ) = data_loader.prepare_data_remaining_time(
                 self.train_df, self.x_word_dict, self.max_case_length)
            # Prepare test examples for remaining time prediction task
            (self.test_token_x, self.test_time_x, self.test_y, _, _
             )= data_loader.prepare_data_remaining_time(
                 self.test_df, self.x_word_dict, self.max_case_length, 
                 self.time_scaler, self.y_scaler, inference=True)   
        end = time.time()
        self.logger.info(f'Total loading time (s): {(end - start)}.')
        self.args = data_loader.return_args()
    
        start = time.time()
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.device)
        #self.logger.info(f'test_token_x device:: {self.test_token_x.device}')
        #tf.debugging.set_log_device_placement(True)
        # Create and train a transformer model
        if self.args.task == 'next_act':
            transformer_model = transformer.get_next_activity_model(
                max_case_length=self.max_case_length,
                vocab_size=self.vocab_size, output_dim=self.num_output)
            transformer_model.compile(
                optimizer=tf.keras.optimizers.Adam(self.args.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.args.model_path, save_weights_only=True, 
                monitor="sparse_categorical_accuracy", mode="max", 
                save_best_only=True)
            transformer_model.fit(
                self.train_token_x, self.train_token_y, epochs=self.args.epochs,
                batch_size=self.args.batch_size, shuffle=True, verbose=2,
                callbacks=[model_checkpoint_callback])
        elif self.args.task == 'next_role':
            transformer_model = transformer.get_next_role_model(
                max_case_length=self.max_case_length,
                vocab_size=self.vocab_size, output_dim=self.num_output)
            transformer_model.compile(
                optimizer=tf.keras.optimizers.Adam(self.args.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.args.model_path, save_weights_only=True, 
                monitor="sparse_categorical_accuracy", mode="max", 
                save_best_only=True)
            transformer_model.fit(
                self.train_token_x, self.train_token_y, epochs=self.args.epochs,
                batch_size=self.args.batch_size, shuffle=True, verbose=2,
                callbacks=[model_checkpoint_callback])
        elif self.args.task == 'next_time':
            transformer_model = transformer.get_next_time_model(
                max_case_length=self.max_case_length, vocab_size=self.vocab_size)
            transformer_model.compile(
                optimizer=tf.keras.optimizers.Adam(self.args.learning_rate),
                loss=tf.keras.losses.LogCosh())
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.args.model_path, save_weights_only=True,
                monitor="loss", save_best_only=True)
            transformer_model.fit([self.train_token_x, self.train_time_x], 
                                  self.train_token_y, epochs=self.args.epochs,
                                  batch_size=self.args.batch_size,
                                  verbose=2, callbacks=[model_checkpoint_callback])
        elif self.args.task == 'rem_time':            
            transformer_model = transformer.get_remaining_time_model(
                max_case_length=self.max_case_length, vocab_size=self.vocab_size)
            transformer_model.compile(
                optimizer=tf.keras.optimizers.Adam(self.args.learning_rate),
                loss=tf.keras.losses.LogCosh())
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.args.model_path, save_weights_only=True,
                monitor="loss", save_best_only=True)
            transformer_model.fit([self.train_token_x, self.train_time_x], 
                                  self.train_token_y, epochs=self.args.epochs, 
                                  batch_size=self.args.batch_size, verbose=2,
                                  callbacks=[model_checkpoint_callback])
            
        end = time.time()
        self.logger.info(f'Total training time (h): {(end - start)/3600}.')
        
        # inference (evaluate)
        start = time.time()
        if self.args.task == 'next_act' or self.args.task == 'next_role':
            y_pred = np.argmax(
                transformer_model.predict(self.test_token_x), axis=1)
            self.test_df['predictions'] = y_pred
            accuracy = metrics.accuracy_score(self.test_token_y, y_pred)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                self.test_token_y, y_pred, average="weighted")
            self.logger.info(f'Accuracy for {self.args.task} prediction: {accuracy}')
            self.logger.info(f'Precision for {self.args.task} prediction: {precision}')
            self.logger.info(f'Recall for {self.args.task} prediction: {recall}')
            self.logger.info(f'F1-score for {self.args.task} prediction: {fscore}')
        elif self.args.task == 'next_time':
            y_pred = transformer_model.predict([self.test_token_x, self.test_time_x])
            _test_y = self.y_scaler.inverse_transform(self.test_y)
            _y_pred = self.y_scaler.inverse_transform(y_pred)
            _test_y = _test_y /3600 /24
            _y_pred = _y_pred /3600 /24
            self.test_df[['Ground_truth_waiting_time', 'Ground_truth_processing_time']] = _test_y
            self.test_df[['predictions_waiting_time', 'predictions_processing_time']] = _y_pred
            # ensure small negative numbers are mapped to zero
            self.test_df['Ground_truth_waiting_time'] = self.test_df['Ground_truth_waiting_time'].apply(lambda x: max(x, 0))
            self.test_df['Ground_truth_processing_time'] = self.test_df['Ground_truth_processing_time'].apply(lambda x: max(x, 0))   
            self.test_df['predictions_waiting_time'] = self.test_df['predictions_waiting_time'].apply(lambda x: max(x, 0))
            self.test_df['predictions_processing_time'] = self.test_df['predictions_processing_time'].apply(lambda x: max(x, 0))  
            mae1 = metrics.mean_absolute_error(_test_y[:, 0], _y_pred[:, 0])
            mae2 = metrics.mean_absolute_error(_test_y[:, 1], _y_pred[:, 1])
            mse1 = metrics.mean_squared_error(_test_y[:, 0], _y_pred[:, 0])
            mse2 = metrics.mean_squared_error(_test_y[:, 1], _y_pred[:, 1])
            mape1 = safe_mape(_test_y[:, 0], _y_pred[:, 0])
            mape2 = safe_mape(_test_y[:, 1], _y_pred[:, 1])           
            self.logger.info(f'Average MAE for waiting time for {self.args.task} prediction: {mae1}')
            self.logger.info(f'Average MAPE for waiting time for {self.args.task} prediction: {mape1}')
            self.logger.info(f'Average MSE for waiting time for {self.args.task} prediction: {mse1}')
            self.logger.info(f'Average MAE for duration time for {self.args.task} prediction: {mae2}')
            self.logger.info(f'Average MAPE for duration time for {self.args.task} prediction: {mape2}')
            self.logger.info(f'Average MSE for duration time for {self.args.task} prediction: {mse2}')
        elif self.args.task == 'rem_time':
            y_pred = transformer_model.predict([self.test_token_x, self.test_time_x])
            _test_y = self.y_scaler.inverse_transform(self.test_y)
            _y_pred = self.y_scaler.inverse_transform(y_pred)
            _test_y = _test_y /3600 /24
            _y_pred = _y_pred /3600 /24
            self.test_df['Ground_truth'] = _test_y
            self.test_df['predictions'] = _y_pred
            # ensure small negative numbers are mapped to zero
            self.test_df['Ground_truth'] = self.test_df['Ground_truth'].apply(lambda x: max(x, 0))
            self.test_df['predictions'] = self.test_df['predictions'].apply(lambda x: max(x, 0))            
            mae = metrics.mean_absolute_error(_test_y, _y_pred)
            mse = metrics.mean_squared_error(_test_y, _y_pred)
            mape = safe_mape(_test_y, _y_pred)
            self.logger.info(f'Average MAE for {self.args.task} prediction: {mae}')
            self.logger.info(f'Average MAPE for {self.args.task} prediction: {mape}')
            self.logger.info(f'Average MSE for {self.args.task} prediction: {mse}')        
        end = time.time()
        self.logger.info(f'Total inference time (s): {(end - start)}.')
        self.logger.info(
                f'Total inference time per prefix (milli s): {(end - start)*1000/len(self.test_df)}.')
        self.test_df.drop('prefix', axis=1, inplace=True)
        if self.args.task == 'next_act' or self.args.task == 'next_role': 
            token_y = list()
            for _y in y_pred:
                key = [k for k, v in self.y_word_dict.items() if v == _y][0]                
                token_y.append(key)
            token_y_arr = np.array(token_y)
            if self.args.task == 'next_act':
                self.test_df['pred_act'] = token_y_arr  
            else:
                self.test_df['pred_role'] = token_y_arr 
        self.test_df.to_csv(self.args.inference_path, index=False)
        for file_path in self.args.temp_files:            
            if os.path.exists(file_path):
                os.remove(file_path)

    def return_args(self):
        return self.args