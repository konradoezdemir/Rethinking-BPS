# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:02:19 2025
@author: Keyvan Amiri Elyasi
to create this scripts in some parts we used the following implementation of the
deep generative LSTM model by Camargo et al.
    https://github.com/AdaptiveBProcess/GenerativeLSTM/
    
"""
import pandas as pd
from datetime import timedelta
import itertools as it
import csv
import numpy as np
from sys import stdout


def append_csv_start_end(data):
    end_start_times = dict()
    for case, group in pd.DataFrame(data).groupby('caseid', as_index=False):
        end_start_times[(case, 'Start')] = (group.start_timestamp.min() - timedelta(microseconds=1))
        end_start_times[(case, 'End')] = (group.end_timestamp.max() + timedelta(microseconds=1))
    new_data = list()
    sorted_data = sorted(data, key=lambda x: x['caseid'])
    for key, group in it.groupby(sorted_data, key=lambda x: x['caseid']):
        trace = list(group)
        trace = create_dummy_events(end_start_times, key, trace)
        new_data.extend(trace)
    return new_data

def create_dummy_events(end_start_times, key, trace):
    for new_event in ['Start', 'End']:
        idx = 0 if new_event == 'Start' else -1
        temp_event = dict()
        temp_event['caseid'] = trace[idx]['caseid']
        temp_event['task'] = new_event
        temp_event['user'] = new_event
        temp_event['end_timestamp'] = end_start_times[(key, new_event)]
        temp_event['start_timestamp'] = end_start_times[(key, new_event)]
        if new_event == 'Start':
            trace.insert(0, temp_event)
        else:
            trace.append(temp_event)
    return trace
    
def split_event_transitions(data):
    temp_raw = list()
    for event in data:
        start_event = event.copy()
        complete_event = event.copy()
        start_event.pop('end_timestamp')
        complete_event.pop('start_timestamp')
        start_event['timestamp'] = start_event.pop('start_timestamp')
        complete_event['timestamp'] = complete_event.pop('end_timestamp')
        start_event['event_type'] = 'start'
        complete_event['event_type'] = 'complete'
        temp_raw.append(start_event)
        temp_raw.append(complete_event)
    return temp_raw
        
def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
        parms:
            log_df: dataframe.
            column: column name.
        Returns:
            index of a categorical attribute pairs.
    """
    temp_list = log_df[[column]].values.tolist()
    subsec_set = {(x[0]) for x in temp_list}
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    for i, _ in enumerate(subsec_set):
        alias[subsec_set[i]] = i + 1
    return alias
    
  

def load_embedded(index, file_path):
    """Loading of the embedded matrices.
        parms:
            index (dict): index of activities or roles.
            filename (str): filename of the matrix file.
        Returns:
            numpy array: array of weights.
    """
    weights = list()
    with open(file_path, 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in filereader:
            cat_ix = int(row[0])
            if index[cat_ix] == row[1].strip():
                weights.append([float(x) for x in row[2:]])
        csvfile.close()
    return np.array(weights)    

#print a csv file from list of lists
def create_file_from_list(index, output_file):
    with open(output_file, 'w') as f:
        for element in index:
            f.write(', '.join(list(map(lambda x: str(x), element))))
            f.write('\n')
        f.close()

#printing process functions
def print_progress(percentage, text):
    stdout.write("\r%s" % text + str(percentage)[0:5] + chr(37) + "...      ")
    stdout.flush()
    
# define columns for vectorizer
def define_columns(add_cols, one_timestamp):
    columns = ['ac_index', 'rl_index', 'dur_norm']
    add_cols = [x+'_norm' if x != 'weekday' else x for x in add_cols]
    columns.extend(add_cols)
    if not one_timestamp:
        columns.extend(['wait_norm'])
    return columns
 
# method to adjust the format of timestamps
def safe_to_datetime(ts):
    try:
        return pd.to_datetime(ts, utc=True)  # Convert while keeping timezone
    except Exception as e:
        print(f"Error parsing timestamp: {ts} -> {e}")
        return pd.NaT  # Return NaT if conversion fails   
    
def align_column_names(df):
    if 'case:concept:name' in df.columns:
        df = df.rename(columns={'case:concept:name': 'case_id'})
    elif 'caseid' in df.columns:
        df = df.rename(columns={'caseid': 'case_id'})
    if 'Activity' in df.columns:
        df = df.rename(columns={'Activity': 'activity_name'})
    elif 'activity' in df.columns:
        df = df.rename(columns={'activity': 'activity_name'})
    elif 'task' in df.columns:
        df = df.rename(columns={'task': 'activity_name'})
    elif 'concept:name' in df.columns:
        df = df.rename(columns={'concept:name': 'activity_name'})
    if 'Resource' in df.columns:
        df = df.rename(columns={'Resource': 'resource'})
    elif 'user' in df.columns:
        df = df.rename(columns={'user': 'resource'})
    elif 'org:resource' in df.columns:
        df = df.rename(columns={'org:resource': 'resource'})
    if 'start_time' in df.columns:
        df = df.rename(columns={'start_time': 'start_timestamp'})
    if 'end_time' in df.columns:
        df = df.rename(columns={'end_time': 'end_timestamp'})
    if 'start:timestamp' in df.columns:
        df = df.rename(columns={'start:timestamp': 'start_timestamp'})
    if 'time:timestamp' in df.columns:
        df = df.rename(columns={'time:timestamp': 'end_timestamp'})
    return df

def safe_mape(y_true, y_pred):
    # Calculate absolute percentage error
    ape = np.abs((y_true - y_pred) / y_true)    
    # Set APE to 0 where ground truth is 0
    ape[y_true == 0] = 0
    return np.mean(ape)