# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 07:30:02 2025
Main python script to train and evaluate one of the supported machine learning
models on the selected event log.
"""

import os
import argparse
import yaml
import logging
import random
import numpy as np
import torch
import tensorflow as tf

from pipeline.transformer_pipeline import TransformerExperiment
from pipeline.pgtnet_pipeline import GraphTransformerExperiment
from pipeline.lstm_pipeline import LSTMExperiment

# Configure logger
logger = logging.getLogger('Rethinking-BPS') 
logger.setLevel(logging.INFO) 
# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    tf.random.set_seed(seed)    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        tf.config.experimental.enable_op_determinism()


def main():    
    # parse arguments and load the relevant cfg file
    parser = argparse.ArgumentParser(
        description='Rethinking-BPS: train-evaluate on real/simulated data.')
    parser.add_argument('--dataset', default='LoanApp',
                        help='Dataset for training-evaluation.') 
    parser.add_argument('--cfg', help='configuration file.')
    parser.add_argument('--sim',
                        help='simulation type for simulated data or real data.')
    parser.add_argument('--model', help='type of machine learning model used.')
    parser.add_argument('--task', help='PPM task at hand.')
    parser.add_argument('--seed', type=int, nargs='+', required=True, 
                        help='List of random seeds')    
    #parser.add_argument('--device', type=int, default=0, help='GPU device id.')
    #parser.add_argument('--thread', type=int, default=4, help='number of threads.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Whether to overwrite existing result or not.')    
    args = parser.parse_args()
    
    # set the device for slurm jobs
    #os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    args.root_path = os.getcwd()    
    # read cfg file, set results path, process data path, and handle the logger
    args.cfg_path = os.path.join(args.root_path, 'cfg',args.cfg)
    with open(args.cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    args.result_path = os.path.join(args.root_path, 'results', args.dataset)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    logger_name = args.dataset+'#'+args.task+'#'+args.sim+'#'+args.model+'.log'
    file_handler = logging.FileHandler(os.path.join(args.result_path, logger_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler) 
    # Preliminaries based on real/simulated data and simulation type
    train_name = cfg['train_name'] + '.csv'       
    test_name = cfg['test_name'] + '.csv'
    args.train_path_org = os.path.join(
        args.root_path, 'data', args.dataset, train_name)
    args.test_path = os.path.join(
        args.root_path, 'data', args.dataset, test_name)
    if args.sim == 'real':
        args.run = 1        
        args.train_path = args.train_path_org       
    else:
        args.run = cfg['simulation_num']
        simulated_name = cfg['simulated_name']
        train_folder = os.path.join(args.root_path, 'data', args.dataset, args.sim)
    # training and evaluation
    for run in range(args.run):
        if args.sim != 'real':
            sim_train_name = simulated_name + str(run) + '.csv'
            args.train_path = os.path.join(train_folder, sim_train_name)
        for seed in args.seed:
            # Set seed
            set_random_seed(seed)
            if args.model == 'transformer':
                exp_obj = TransformerExperiment(
                    args=args, cfg=cfg, run=run, seed=seed, logger=logger)
                exp_obj.execute_pipeline()
            elif args.model == 'lstm':           
                exp_obj = LSTMExperiment(
                    args=args, cfg=cfg, run=run, seed=seed, logger=logger)
                exp_obj.execute_pipeline()
            elif args.model == 'graph_transformer':
                exp_obj = GraphTransformerExperiment(
                    args=args, cfg=cfg, run=run, seed=seed, logger=logger)

        
if __name__ == '__main__':
    main()   

            

