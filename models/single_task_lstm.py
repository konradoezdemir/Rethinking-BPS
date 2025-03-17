import os
import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report)

 

 
class MyModel(nn.Module):
    def __init__(self, ac_weights, rl_weights, lstm_size, act_dim, role_dim, 
                 time_dim, task_name):
        super(MyModel, self).__init__()
        self.task_name = task_name
 
        if self.task_name == 'next_act':
            # BACKBONE
            # define pretrained activity embedding layer
            self.embedding = nn.Embedding(ac_weights.shape[0],
                                ac_weights.shape[1],)
            self.embedding.weight = torch.nn.Parameter(torch.from_numpy(ac_weights))
            self.embedding.weight.requires_grad = False
            # individual first LSTM layer per task
            # activity layer 1, and subsequent batch norm and dropout
            self.lstm1_act = nn.LSTM(input_size=act_dim, hidden_size=lstm_size, batch_first=True)
            self.batch_norm_act = nn.BatchNorm1d(lstm_size)
            self.dropout_act = nn.Dropout(p=0.2)
            # activity layer 2
            self.lstm2_act = nn.LSTM(lstm_size, lstm_size, batch_first=True)
            # Fully connected activity head
            self.fc_act = nn.Linear(lstm_size, ac_weights.shape[0]) 
        elif self.task_name == 'next_role':
            # define pretrained role embedding layer
            self.embedding = nn.Embedding(rl_weights.shape[0],
                                rl_weights.shape[1],)
            self.embedding.weight = torch.nn.Parameter(torch.from_numpy(rl_weights))
            self.embedding.weight.requires_grad = False
            # role layer 1, and subsequent batch norm and dropout
            self.lstm1_role = nn.LSTM(input_size=role_dim, hidden_size=lstm_size, batch_first=True)
            self.batch_norm_role = nn.BatchNorm1d(lstm_size)
            self.dropout_role = nn.Dropout(p=0.2)
            # role layer 2
            self.lstm2_role = nn.LSTM(lstm_size, lstm_size, batch_first=True)
            # Fully connected role head
            self.fc_role = nn.Linear(lstm_size, rl_weights.shape[0])  
        elif self.task_name == 'next_time':
            # time layer 1, and subsequent batch norm and dropout
            self.lstm1_time = nn.LSTM(input_size=time_dim, hidden_size=lstm_size, batch_first=True)
            self.batch_norm_time = nn.BatchNorm1d(lstm_size)
            self.dropout_time = nn.Dropout(p=0.2)
            # time layer 2
            self.lstm2_time = nn.LSTM(lstm_size, lstm_size, batch_first=True)
            # Fully connected time head
            self.fc_time = nn.Linear(lstm_size, time_dim)

 
    def forward(self, ac_input, rl_input, t_input):
        if self.task_name == 'next_act':
            ac_embedded = self.embedding(ac_input)
            # activity layer 1
            lstm1_out_act, (h1_act,c1_act) = self.lstm1_act(ac_embedded)
            lstm1_out_act = self.dropout_act(lstm1_out_act)
            # transpose dimensions of features and sequence
            lstm1_out_act = lstm1_out_act.transpose(1, 2) 
            batch_norm_out_act = self.batch_norm_act(lstm1_out_act)
            batch_norm_out_act = batch_norm_out_act.transpose(1, 2)
            # activity layer 2
            lstm2_act_out, _ = self.lstm2_act(batch_norm_out_act,(h1_act,c1_act))
            lstm2_act_out = self.dropout_act(lstm2_act_out)
            # Fully connected act
            act_output = self.fc_act(lstm2_act_out[:, -1, :])
            act_output = F.softmax(act_output, dim=-1)
            return act_output
        elif self.task_name == 'next_role':
            rl_embedded = self.embedding(rl_input)
            # role layer 1
            lstm1_out_role, (h1_role,c1_role) = self.lstm1_role(rl_embedded)
            lstm1_out_role = self.dropout_role(lstm1_out_role)
            # transpose dimensions of features and sequence
            lstm1_out_role = lstm1_out_role.transpose(1, 2)
            batch_norm_out_role = self.batch_norm_role(lstm1_out_role)
            batch_norm_out_role = batch_norm_out_role.transpose(1, 2)
            # role layer 2
            lstm2_role_out, _ = self.lstm2_role(batch_norm_out_role,(h1_role,c1_role))
            lstm2_role_out = self.dropout_role(lstm2_role_out)
            # Fully connected role
            role_output = self.fc_role(lstm2_role_out[:, -1, :])
            role_output = F.softmax(role_output, dim=-1)
            return role_output
        elif self.task_name == 'next_time':
            # time layer 1
            lstm1_out_time, (h1_time,c1_time) = self.lstm1_time(t_input)
            lstm1_out_time = self.dropout_time(lstm1_out_time)
            lstm1_out_time = lstm1_out_time.transpose(1, 2)
            # transpose dimensions of features and sequence
            batch_norm_out_time = self.batch_norm_time(lstm1_out_time)
            batch_norm_out_time = batch_norm_out_time.transpose(1, 2) 
            # time layer 2
            lstm2_time_out, _ = self.lstm2_time(batch_norm_out_time,(h1_time,c1_time))
            lstm2_time_out = self.dropout_time(lstm2_time_out)
            # Fully connected time
            time_output = self.fc_time(lstm2_time_out[:, -1, :])
            # to avoid negative predictions it might be possible to introduce
            # some activation functions like ReLU or softplus. However, it may
            # degrade the capacity o the network. Therefore, we do not use
            # any activation function, and adjust the prediction in a post-hoc 
            # manner. 
            #time_output = F.relu(time_output)             
            #time_output = torch.log1p(torch.exp(time_output))
            # time_output = F.softplus(time_output)           
            return time_output
    
 
def train_model(model, dataloader, test_dataloader, val_dataloader,
                num_epochs=10, optimizer=None, scheduler=None,
                task_name='next_act', prediction_method='argmax', 
                early_patience=None, min_delta=None,
                ac_index=None, rl_index=None, time_dim=2, dataset_name=None,
                logger=None, checkpoint_path=None):
    
    cuda_available = torch.cuda.is_available()
    # Check where the model is located
    device = next(model.parameters()).device
    print(f"Model is training on: {device} (CUDA Available: {cuda_available})")
    if logger:
        logger.info(f"Model is training on: {device} (CUDA Available: {cuda_available})")
    if device.type != "cuda":
        raise ValueError(f"Invalid device detected: {device}. Expected 'cpu' or 'cuda'.")  
    
    
    # Training phase
    if task_name == 'next_act' or task_name == 'next_role':
        criterion = nn.CrossEntropyLoss()
    elif task_name == 'next_time':
        criterion = nn.L1Loss()        
    current_patience = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        for batch in dataloader:
            ac_input_batch, rl_input_batch, t_input_batch, target_act_batch, \
                target_role_batch, target_time_batch = batch
            output_batch = model(ac_input_batch, rl_input_batch, t_input_batch)
            #logger.info(f'ac_input_batch shape: {ac_input_batch.shape}')
            #logger.info(f'rl_input_batch shape: {rl_input_batch.shape}')
            #logger.info(f't_input_batch shape: {t_input_batch.shape}')
            #logger.info(f'target_act_batch shape: {target_act_batch.shape}')
            #logger.info(f'target_role_batch shape: {target_role_batch.shape}')
            #logger.info(f'target_time_batch shape: {target_time_batch.shape}')
            #logger.info(f'output_batch shape: {output_batch.shape}')
            if task_name == 'next_act':
                target = target_act_batch
            elif task_name == 'next_role':
                target = target_role_batch
            elif task_name == 'next_time':
                target = target_time_batch 
            loss = criterion(output_batch, target) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss for {task_name}: {average_loss}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss for {task_name}: {average_loss}') 
        # perform validation loop
        if val_dataloader is not None:
            average_validation_loss = test_model(
                model, val_dataloader, criterion, task_name, dataset_name, 
                prediction_method=prediction_method, ac_index=ac_index, 
                rl_index=rl_index, time_dim=time_dim, inference=False, logger=logger)            
            # early stopping
            if average_validation_loss < best_val_loss * (1 - min_delta):                
                best_val_loss = average_validation_loss
                current_patience = 0
                checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'best_valid_loss': best_val_loss            
                }
                torch.save(checkpoint, checkpoint_path)
            else:
                current_patience += 1
                if current_patience >= early_patience:
                    print("Early stopping: Val loss has not improved for {} epochs.".format(early_patience))
                    break        
        # if we have a learning rate scheduler
        if scheduler is not None:
            scheduler.step()

    
def test_model(model, dataloader, criterion, task_name, dataset_name, 
               prediction_method='argmax', ac_index=None, rl_index=None,
               time_dim=2, norm_dict=None, inference=False, case_id_lst=None, 
               prefix_length_lst=None, logger=None, result_path=None):
    """
    test the model on validation or test data
    """
    test_loss = 0.0
    model.eval()  # Set the model to evaluation mode    
    if inference:
        if task_name == 'next_act':
            all_results = {'Predicted_Act': [], 'GroundTruth_Act': []}
            reverse_ac_index = {v: k for k, v in ac_index.items()}
        elif task_name == 'next_role':
            all_results = {'Predicted_Role': [], 'GroundTruth_Role': []}
            reverse_rl_index = {v: k for k, v in rl_index.items()}
        elif task_name == 'next_time':
            if time_dim == 1:
                all_results = {'Predicted_dur': [], 'GroundTruth_dur': []}  
            else:
                all_results = {'Predicted_dur': [], 'GroundTruth_dur': [],
                               'Predicted_wait': [], 'GroundTruth_wait': []} 
    
    with torch.no_grad():
        for test_batch in dataloader:
            test_ac_input_batch, test_rl_input_batch, test_t_input_batch, \
            test_target_act_batch, test_target_role_batch, \
                test_target_time_batch = test_batch 
            test_output_batch = model(
                test_ac_input_batch, test_rl_input_batch, test_t_input_batch) 
            #print(test_output_batch.shape)
            if task_name == 'next_act':
                target = test_target_act_batch
            elif task_name == 'next_role':
                target = test_target_role_batch
            elif task_name == 'next_time':
                target = test_target_time_batch 
            test_loss += criterion(test_output_batch, target).item()            
            # Collect predicted and ground truth values for each example in the batch
            if inference:
                for i in range(len(test_ac_input_batch)):
                    if task_name == 'next_act' or task_name == 'next_role':
                        if prediction_method=='argmax':
                            predicted_idx = np.argmax(test_output_batch[i].detach().cpu().numpy())
                        elif prediction_method=='random':
                            predicted_idx = np.random.choice(
                                len(test_output_batch[i].detach().cpu().numpy()),
                                p=test_output_batch[i].detach().cpu().numpy())
                        else:
                            raise ValueError('Unsupported prediction method. Choose from: argmax, random.')
                        if task_name == 'next_act':                            
                            predicted_activity = reverse_ac_index.get(predicted_idx)
                            target_activity = reverse_ac_index.get(
                                np.argmax(test_target_act_batch[i].detach().cpu().numpy()))
                            all_results['Predicted_Act'].append(predicted_activity)
                            all_results['GroundTruth_Act'].append(target_activity)
                        else:
                            predicted_role = reverse_rl_index.get(predicted_idx)
                            target_role = reverse_rl_index.get(
                                np.argmax(test_target_role_batch[i].detach().cpu().numpy()))
                            all_results['Predicted_Role'].append(predicted_role)
                            all_results['GroundTruth_Role'].append(target_role)
                    elif task_name == 'next_time':
                        if time_dim == 1:
                            pred_dur = test_output_batch[i][0].detach().cpu().numpy()
                            real_dur = test_target_time_batch[i][0].detach().cpu().numpy()
                            # enforce positive predictions
                            pred_dur = max(0, pred_dur)
                            if bool(norm_dict):
                                dur_norm = norm_dict['dur']
                                norm_type = next(iter(dur_norm))
                                if norm_type == 'max_value':
                                    max_val = dur_norm['max_value']
                                    pred_dur = pred_dur*max_val/3600/24
                                    real_dur = real_dur*max_val/3600/24
                                    all_results['Predicted_dur'].append(pred_dur)
                                    all_results['GroundTruth_dur'].append(real_dur)
                                else:
                                    logger.info(f'adjust the code for {norm_type} normalization.')
                                    all_results['Predicted_dur'].append(pred_dur)
                                    all_results['GroundTruth_dur'].append(real_dur)   
                            else:
                                all_results['Predicted_dur'].append(pred_dur)
                                all_results['GroundTruth_dur'].append(real_dur)                               
                        else:
                            pred_dur = test_output_batch[i][0].detach().cpu().numpy()
                            real_dur = test_target_time_batch[i][0].detach().cpu().numpy()
                            pred_wait = test_output_batch[i][1].detach().cpu().numpy()
                            real_wait = test_target_time_batch[i][1].detach().cpu().numpy()
                            # enforce positive predictions
                            pred_dur = max(0, pred_dur)
                            pred_wait = max(0, pred_wait)
                            if bool(norm_dict):
                                dur_norm = norm_dict['dur']
                                wait_norm = norm_dict['wait']
                                norm_type = next(iter(dur_norm))
                                if norm_type == 'max_value':
                                    max_val = dur_norm['max_value']
                                    pred_dur = pred_dur*max_val/3600/24
                                    real_dur = real_dur*max_val/3600/24
                                    all_results['Predicted_dur'].append(pred_dur)
                                    all_results['GroundTruth_dur'].append(real_dur)
                                else:
                                    logger.info(f'adjust the code for {norm_type} normalization.')
                                    all_results['Predicted_dur'].append(pred_dur)
                                    all_results['GroundTruth_dur'].append(real_dur) 
                                norm_type = next(iter(wait_norm))
                                if norm_type == 'max_value':
                                    max_val = wait_norm['max_value']
                                    pred_wait = pred_wait*max_val/3600/24
                                    real_wait = real_wait*max_val/3600/24
                                    all_results['Predicted_wait'].append(pred_wait)
                                    all_results['GroundTruth_wait'].append(real_wait)
                                else:
                                    logger.info(f'adjust the code for {norm_type} normalization.')
                                    all_results['Predicted_wait'].append(pred_wait)
                                    all_results['GroundTruth_wait'].append(real_wait)
                            else:
                                all_results['Predicted_dur'].append(pred_dur)
                                all_results['GroundTruth_dur'].append(real_dur) 
                                all_results['Predicted_wait'].append(pred_wait)
                                all_results['GroundTruth_wait'].append(real_wait)
        num_test_batches = len(dataloader)
        test_loss /= num_test_batches 
    if not inference:
        print(f'Validation - Loss for {task_name}: {test_loss}')
        logger.info(f'Validation - Loss for {task_name}: {test_loss}')
    else:
        print(f'Test - Loss for {task_name}: {test_loss}')
        logger.info(f'Test - Loss for {task_name}: {test_loss}')
        # Create a pandas DataFrame from the collected results
        results_df = pd.DataFrame(all_results)
        if task_name == 'next_time':
            if time_dim == 1:
                results_df[['Predicted_dur','GroundTruth_dur']] = results_df[['Predicted_dur','GroundTruth_dur']].astype(float)
            else:
                results_df[['Predicted_dur','GroundTruth_dur','Predicted_wait', 'GroundTruth_wait']] = results_df[['Predicted_dur','GroundTruth_dur','Predicted_wait', 'GroundTruth_wait']].astype(float)
        #logger.info(f'number of test instances: {len(results_df)}')
        #logger.info(f'number of case_ids: {len(case_id_lst)}')
        #logger.info(f'number of pls: {len(prefix_length_lst)}')
        results_df['case_id'] = case_id_lst
        results_df['k'] = prefix_length_lst
        results_df.to_csv(result_path, index=False)
        print(f'Results have been saved to {result_path}')
        logger.info(f'Results have been saved to {result_path}')
        compute_overal_metrics(results_df, dataset_name, task_name, logger,
                               time_dim=time_dim)
    return test_loss # required for early stopping

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def perform_evaluation(args, params, test_dataloader=None, test_dataset=None, 
                       act_dim=None, role_dim=None, time_dim=None, ac_index=None, 
                       rl_index=None, ac_weights=None, rl_weights=None,
                       rem_time_dict=None, case_id_lst=None, prefix_length_lst=None,
                       logger=None):
    dataset_name = args.dataset
    task_name = args.task
    lstm_size = params['l_size']
    prediction_method = params['prediction_method']
    norm_dict = params['scale_args']
    result_path = args.inference_path
    if task_name == 'rem_time':
        if torch.cuda.is_available():
            slurm_gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0")
            device = f'cuda:{slurm_gpu_id}'
        else:
            device = 'cpu'    
        act_model = MyModel(ac_weights=ac_weights, rl_weights=rl_weights,
                            lstm_size=lstm_size, act_dim=act_dim,
                            role_dim=role_dim, time_dim=time_dim,
                            task_name='next_act')
        act_model = to_cuda(act_model)
        act_checkpoint = torch.load(args.act_model_path, map_location=device)
        act_model.load_state_dict(act_checkpoint['model_state_dict'])
        act_model.double()
        act_model.eval()
        
        time_model = MyModel(ac_weights=ac_weights, rl_weights=rl_weights,
                            lstm_size=lstm_size, act_dim=act_dim,
                            role_dim=role_dim, time_dim=time_dim,
                            task_name='next_time')
        time_model = to_cuda(time_model)
        time_checkpoint = torch.load(args.time_model_path, map_location=device)
        time_model.load_state_dict(time_checkpoint['model_state_dict'])
        time_model.double()
        time_model.eval()
        
        role_model = MyModel(ac_weights=ac_weights, rl_weights=rl_weights,
                            lstm_size=lstm_size, act_dim=act_dim,
                            role_dim=role_dim, time_dim=time_dim,
                            task_name='next_role')
        role_model = to_cuda(role_model)
        role_checkpoint = torch.load(args.role_model_path, map_location=device)
        role_model.load_state_dict(role_checkpoint['model_state_dict'])
        role_model.double()
        role_model.eval()
        
        rem_time_halluciantion(args, params, test_dataset=test_dataset,
                               act_model=act_model, time_model=time_model,
                               role_model=role_model, ac_index=ac_index,
                               rem_time_dict=rem_time_dict, case_id_lst=case_id_lst,
                               prefix_length_lst=prefix_length_lst, logger=logger)     
    else:       
        model = MyModel(ac_weights=ac_weights, rl_weights=rl_weights,
                        lstm_size=lstm_size, act_dim=act_dim, role_dim=role_dim,
                        time_dim=time_dim, task_name=task_name)
        model = to_cuda(model)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.double()
        if task_name == 'next_act' or task_name == 'next_role':
            criterion = nn.CrossEntropyLoss()
        elif task_name == 'next_time':
            criterion = nn.L1Loss() 
        _ = test_model(model, test_dataloader, criterion, task_name, 
                       dataset_name, prediction_method=prediction_method,
                       ac_index=ac_index, rl_index=rl_index, time_dim=time_dim,
                       norm_dict=norm_dict, inference=True, case_id_lst=case_id_lst,
                       prefix_length_lst=prefix_length_lst, logger=logger,
                       result_path=result_path)


def rem_time_halluciantion(args, params, test_dataset=None, act_model=None, 
                           time_model=None, role_model=None, ac_index=None, 
                           rem_time_dict=None, case_id_lst=None,
                           prefix_length_lst=None, logger=None):
    one_timestamp = params['one_timestamp']
    prediction_method = params['prediction_method']
    norm_dict = params['scale_args']
    max_length = int(params['max_length']) + 2
    # reverse activity index to check reaching 'end'
    reverse_ac_index = {v: k for k, v in ac_index.items()}
    all_results = {'case_id':[], 'k': [], 'Predicted_rem_time': [],
                   'Ground_truth_rem_time':[], 'Predicted_suffix': []}  
    prefix_idx = 0
    with torch.no_grad():
        for prefix in test_dataset:
            pl = prefix_length_lst[prefix_idx]
            case_id = case_id_lst[prefix_idx]
            idx_pred = 0
            passed_time = 0
            predicted_suffix = []
            current_act = 'start'
            ac_inp, rl_inp, t_inp, _, _, _ = prefix
            while ((idx_pred+pl < max_length) and current_act != 'end'):
                ac_inp_temp = ac_inp.unsqueeze(0)
                rl_inp_temp = rl_inp.unsqueeze(0) 
                t_inp_temp = t_inp.unsqueeze(0)    
                # get next event's activty, role indices, and its timestamp(s)
                ac_out = act_model(ac_inp_temp, rl_inp_temp, t_inp_temp)
                rl_out = role_model(ac_inp_temp, rl_inp_temp, t_inp_temp)
                t_out = time_model(ac_inp_temp, rl_inp_temp, t_inp_temp) 
                if prediction_method=='argmax':
                    pred_act_idx = np.argmax(ac_out[0].detach().cpu().numpy())
                    pred_role_idx = np.argmax(rl_out[0].detach().cpu().numpy())
                elif prediction_method=='random':
                    pred_act_idx = np.random.choice(
                        len(ac_out[0].detach().cpu().numpy()),
                        p=ac_out[0].detach().cpu().numpy())
                    pred_role_idx = np.random.choice(
                        len(rl_out[0].detach().cpu().numpy()),
                        p=rl_out[0].detach().cpu().numpy())         
                # update current activity and predicted suffix
                current_act = reverse_ac_index.get(pred_act_idx)
                predicted_suffix.append(current_act)
                if current_act != 'end': 
                    # otherwise passed time remains constant
                    if one_timestamp:
                        pred_dur = max(0, t_out.detach().cpu().numpy().item())
                        # update remaining time prediction
                        if bool(norm_dict):
                            dur_norm = norm_dict['dur']
                            norm_type = next(iter(dur_norm))
                            if norm_type == 'max_value':
                                max_val = dur_norm['max_value']
                                pred_dur = pred_dur*max_val/3600/24
                            else:
                                logger.info(f'adjust the code for {norm_type} normalization.')  
                        passed_time += pred_dur
                    else:
                        pred_dur, pred_wait = t_out.detach().cpu().numpy().flatten()
                        pred_dur = max(0, pred_dur)
                        pred_wait = max(0, pred_wait)
                        # update remaining time prediction
                        if bool(norm_dict):
                            dur_norm = norm_dict['dur']
                            wait_norm = norm_dict['wait']
                            norm_type = next(iter(dur_norm))
                            if norm_type == 'max_value':
                                max_val = dur_norm['max_value']
                                pred_dur = pred_dur*max_val/3600/24
                            else:
                                logger.info(f'adjust the code for {norm_type} normalization.')
                            norm_type = next(iter(wait_norm))
                            if norm_type == 'max_value':
                                max_val = wait_norm['max_value']
                                pred_wait = pred_wait*max_val/3600/24
                            else:
                                logger.info(f'adjust the code for {norm_type} normalization.')                      
                        passed_time += (pred_dur+pred_wait)
                    # update input tensors for next prediction
                    ac_inp = torch.roll(ac_inp, shifts=-1)
                    ac_inp[-1] = torch.tensor(
                        pred_act_idx, dtype=ac_inp.dtype, device=ac_inp.device)
                    rl_inp = torch.roll(rl_inp, shifts=-1)
                    rl_inp[-1] = torch.tensor(
                        pred_role_idx, dtype=rl_inp.dtype, device=rl_inp.device)
                    if one_timestamp:
                        t_inp = torch.roll(t_inp, shifts=-1, dims=0)
                        t_inp[-1, 0] = torch.tensor(
                            pred_dur, dtype=torch.float64, device=t_inp.device)
                    else:
                        t_inp = torch.roll(t_inp, shifts=-1, dims=0)
                        t_inp[-1, :] = torch.tensor(
                            [pred_dur, pred_wait], dtype=torch.float64,
                            device=t_inp.device)              
                # update prediction index
                idx_pred +=1
            all_results['case_id'].append(case_id)
            all_results['k'].append(pl)
            all_results['Predicted_rem_time'].append(passed_time)
            real_rem_time = rem_time_dict.get((case_id, pl)) 
            all_results['Ground_truth_rem_time'].append(real_rem_time)
            all_results['Predicted_suffix'].append(predicted_suffix)           
            prefix_idx +=1
    results_df = pd.DataFrame(all_results)
    results_df[['Predicted_rem_time','Ground_truth_rem_time']] = results_df[['Predicted_rem_time','Ground_truth_rem_time']].astype(float)
    results_df.to_csv(args.inference_path, index=False)
    print(f'Results have been saved to {args.inference_path}')
    logger.info(f'Results have been saved to {args.inference_path}')
    compute_overal_metrics(results_df, args.dataset, args.task, logger) 
            
        
def compute_overal_metrics(results_df, dataset_name, task_name, logger, time_dim=None):
    if task_name == 'next_act':
        predicted_column = 'Predicted_Act'
        label_column = 'GroundTruth_Act'
    elif task_name == 'next_role':
        predicted_column = 'Predicted_Role'
        label_column = 'GroundTruth_Role' 
    elif task_name == 'next_time':
        if time_dim == 1:
            predicted_column = 'Predicted_dur'
            label_column = 'GroundTruth_dur'
        else:
            predicted_column1 = 'Predicted_dur'
            label_column1 = 'GroundTruth_dur' 
            predicted_column2 = 'Predicted_wait'
            label_column2 = 'GroundTruth_wait' 
    elif task_name == 'rem_time':
        predicted_column = 'Predicted_rem_time'
        label_column = 'Ground_truth_rem_time'
    initial_num_test = len(results_df)
    logger.info(f'Number of test examples for {task_name} prediction: {initial_num_test}')
    if (task_name == 'next_act' or task_name == 'next_role' or
        task_name == 'rem_time' or (task_name == 'next_time' and time_dim==1)):
        results_df = results_df.dropna(subset=[predicted_column, label_column])
    else:
        results_df = results_df.dropna(subset=[predicted_column1, label_column1,
                                               predicted_column2, label_column2])
    logger.info(f'Number of NaN values in inference dataframe: {initial_num_test-len(results_df)}')
    # classification metrics        
    if task_name == 'next_act' or task_name == 'next_role':
        # Compute overall result for next activity prediction
        accuracy = accuracy_score(results_df[label_column], 
                                  results_df[predicted_column])
        precision = precision_score(
            results_df[label_column], results_df[predicted_column],
            average='weighted')
        recall = recall_score(
            results_df[label_column], results_df[predicted_column],
            average='weighted')
        f1 = f1_score(
            results_df[label_column], results_df[predicted_column],
            average='weighted')
        print(f'Accuracy for {task_name} prediction: {accuracy}')
        logger.info(f'Accuracy for {task_name} prediction: {accuracy}')
        print(f'Precision for {task_name} prediction: {precision}')
        logger.info(f'Precision for {task_name} prediction: {precision}')
        print(f'Recall for {task_name} prediction: {recall}')
        logger.info(f'Recall for {task_name} prediction: {recall}')
        print(f'F1-score for {task_name} prediction: {f1}')
        logger.info(f'F1-score for {task_name} prediction: {f1}')
        # Recall and F-score are being set to 0.0 in labels with no true samples. 
        report = classification_report(
            results_df[label_column], results_df[predicted_column], 
            digits=4, zero_division=0.0)
        logger.info('Classification report for {task_name} prediction:')
        logger.info(report)
    # regression metrics   
    if task_name == 'next_time' or task_name == 'rem_time':
        if time_dim == 1 or task_name == 'rem_time':
            results_df['MAE'] = results_df.apply(
                lambda row: mean_absolute_error(
                    [row[label_column]], [row[predicted_column]]), axis=1)
            results_df['MAPE'] = results_df.apply(
                lambda row: 0 if row[label_column] == 0 else abs(
                    (row[label_column] - row[predicted_column]) / row[label_column]
                    ), axis=1)
            results_df['MSE'] = results_df.apply(
                lambda row: mean_squared_error(
                    [row[label_column]], [row[predicted_column]]), axis=1)
            average_mae = results_df['MAE'].mean()
            average_mape = results_df['MAPE'].mean()
            average_mse = results_df['MSE'].mean()
            print(f'Average MAE for {task_name} prediction: {average_mae}')
            logger.info(f'Average MAE for {task_name} prediction: {average_mae}')
            print(f'Average MAPE for {task_name} prediction: {average_mape}')
            logger.info(f'Average MAPE for {task_name} prediction: {average_mape}')
            print(f'Average MSE for {task_name} prediction: {average_mse}')
            logger.info(f'Average MSE for {task_name} prediction: {average_mse}')
        else:
            results_df['MAE1'] = results_df.apply(
                lambda row: mean_absolute_error(
                    [row[label_column1]], [row[predicted_column1]]), axis=1)
            results_df['MAE2'] = results_df.apply(
                lambda row: mean_absolute_error(
                    [row[label_column2]], [row[predicted_column2]]), axis=1)
            results_df['MAPE1'] = results_df.apply(
                lambda row: 0 if row[label_column1] == 0 else abs(
                    (row[label_column1] - row[predicted_column1]) / row[label_column1]
                    ), axis=1)
            results_df['MAPE2'] = results_df.apply(
                lambda row: 0 if row[label_column2] == 0 else abs(
                    (row[label_column2] - row[predicted_column2]) / row[label_column2]
                    ), axis=1)
            results_df['MSE1'] = results_df.apply(
                lambda row: mean_squared_error(
                    [row[label_column1]], [row[predicted_column1]]), axis=1)
            results_df['MSE2'] = results_df.apply(
                lambda row: mean_squared_error(
                    [row[label_column2]], [row[predicted_column2]]), axis=1)
            average_mae1 = results_df['MAE1'].mean()
            average_mape1 = results_df['MAPE1'].mean()
            average_mse1 = results_df['MSE1'].mean()
            average_mae2 = results_df['MAE2'].mean()
            average_mape2 = results_df['MAPE2'].mean()
            average_mse2 = results_df['MSE2'].mean()
            print(f'Average MAE for duration time for {task_name} prediction: {average_mae1}')
            logger.info(f'Average MAE for duration time for {task_name} prediction: {average_mae1}')
            print(f'Average MAPE for duration time for {task_name} prediction: {average_mape1}')
            logger.info(f'Average MAPE for duration time for {task_name} prediction: {average_mape1}')
            print(f'Average MSE for duration time for {task_name} prediction: {average_mse1}')
            logger.info(f'Average MSE for duration time for {task_name} prediction: {average_mse1}')
            print(f'Average MAE for waiting time for {task_name} prediction: {average_mae2}')
            logger.info(f'Average MAE for waiting time for {task_name} prediction: {average_mae2}')
            print(f'Average MAPE for waiting time for {task_name} prediction: {average_mape2}')
            logger.info(f'Average MAPE for waiting time for {task_name} prediction: {average_mape2}')
            print(f'Average MSE for waiting time for {task_name} prediction: {average_mse2}')
            logger.info(f'Average MSE for waiting time for {task_name} prediction: {average_mse2}')   