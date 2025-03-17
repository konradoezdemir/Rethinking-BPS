# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:02:35 2025
"""
import os
import re
import pandas as pd
import numpy as np


def get_search_str(task, model):
    if task == 'next_act':
        search_str = 'Accuracy for next_act prediction'
    elif task == 'next_role':
        search_str = 'Accuracy for next_role prediction'  
    elif task == 'next_time':
        search_str = 'Average MAE for duration time for next_time prediction'
        search_str2 = 'Average MAE for waiting time for next_time prediction'        
    elif task == 'rem_time':
        search_str = 'Average MAE for rem_time prediction'
    if task == 'next_time': 
        return search_str, search_str2
    else:
        return search_str

def extract_metric_values(path, search_str, trials):
    pattern = fr"{re.escape(search_str)}:\s*([-+]?\d*\.\d+|\d+)"
    metric_vals = []
    with open(path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the number and convert it to float
                metric_vals.append(float(match.group(1)))
            #print(search_str)
            #print(line)
    #print(len(metric_vals))
    if len(metric_vals) == trials:    
        mean, std = np.mean(metric_vals), np.std(metric_vals)
        complete = True
    elif len(metric_vals) > trials:
        metric_vals = metric_vals[0-trials:]
        mean, std = np.mean(metric_vals), np.std(metric_vals)
        complete = True
    else:
        mean, std = None, None
        complete = False
    return mean, std, complete
       

def main():   
    datasets = ['P2P', 'Production', 'ConsultaDataMining', 'Confidential_1000',
                'Confidential_2000', 'cvs_pharmacy', 'BPIC_2012_W',
                'BPIC_2017_W']
    tasks = ['next_act', 'next_time', 'next_role', 'rem_time']
    simulators = ['AgentSimulator', 'DeepSimulator', 'real', 'Simod']
    models = ['lstm', 'transformer']
    
    root_path = os.getcwd()
    eval_dict = {'dataset': [], 'task': [], 'simulation': [], 'model': [], 
                 'finished': [], 'complete': [], 'avg': [], 'std': []} 
    df_path = os.path.join(root_path, 'results', 'overall_results.csv')
    for dataset in datasets:
        res_path = os.path.join(root_path, 'results', dataset)
        for task in tasks:
            for sim in simulators:
                for model in models:                            
                    log_name = dataset+'#'+task+'#'+sim+'#'+model+'.log'
                    log_path = os.path.join(res_path, log_name)
                    if os.path.exists(log_path):
                        if task == 'next_time': 
                            search_str, search_str2  = get_search_str(task, model)
                            mean, std, complete = extract_metric_values(log_path, search_str, 10)
                            mean2, std2, complete2 = extract_metric_values(log_path, search_str2, 10)
                            eval_dict['dataset'].append(dataset)
                            eval_dict['task'].append('next_duration')
                            eval_dict['task'].append('next_waiting')
                            eval_dict['simulation'].append(sim)
                            eval_dict['model'].append(model)
                            eval_dict['finished'].append(True)
                            eval_dict['finished'].append(True)
                            eval_dict['complete'].append(complete)
                            eval_dict['complete'].append(complete2)
                            eval_dict['avg'].append(mean)
                            eval_dict['avg'].append(mean2)
                            eval_dict['std'].append(std)
                            eval_dict['std'].append(std2)
                        else:
                            search_str = get_search_str(task, model)
                            mean, std, complete = extract_metric_values(log_path, search_str, 10)
                            eval_dict['task'].append(task)
                            eval_dict['finished'].append(True)
                            eval_dict['complete'].append(complete)
                            eval_dict['avg'].append(mean)
                            eval_dict['std'].append(std)
                    else:
                        eval_dict['task'].append(task)
                        eval_dict['finished'].append(False)
                        eval_dict['complete'].append(False)
                        eval_dict['avg'].append('NA')
                        eval_dict['std'].append('NA')
                    eval_dict['dataset'].append(dataset)
                    eval_dict['simulation'].append(sim)
                    eval_dict['model'].append(model)
    # convert dict to dataframe
    eval_df = pd.DataFrame(eval_dict)
    eval_df.to_csv(df_path, index=False)          
    

if __name__ == '__main__':
    main()