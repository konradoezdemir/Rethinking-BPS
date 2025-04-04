"""
This script is part of the following github repository, which is used in our implementation.
    https://github.com/AdaptiveBProcess/GenerativeLSTM/
"""
import random
import itertools
import math
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Reshape
from pre_process.lstm_utils import create_file_from_list
import tensorflow as tf




def training_model(parameters, log, ac_index, index_ac, rl_index, index_rl,
                   ac_emb_path, rl_emb_path):
    
    #tf.config.set_visible_devices([], 'GPU')
    #print(tf.__version__)
    #print(torch.version.cuda)
    
    # Main method of the embedding training module.
    # Define the number of dimensions as the 4th root of the # of categories
    dim_number = math.ceil(
        len(list(itertools.product(*[list(ac_index.items()),
                                     list(rl_index.items())])))**0.25) 
    
    #TODO: embedding training is not possible for BPIC15 event logs: additional 
    #peprocessing is done in original paper.
    #the same limitation holds for sepsis, and BPIC12 evet logs
    
    # switch to CPU due to unresolvable problem in the original code.
    with tf.device('/CPU:0'):
        ac_weights, rl_weights = train_embedded(log, ac_index, rl_index, dim_number)
    create_file_from_list(reformat_matrix(index_ac, ac_weights), ac_emb_path)
    create_file_from_list(reformat_matrix(index_rl, rl_weights), rl_emb_path)
    # Get the list of available GPUs
    #gpus = tf.config.list_physical_devices('GPU')
    # Enable GPU
    #if gpus:
        #tf.config.set_visible_devices(gpus, 'GPU')
    
def train_embedded(log_df, ac_index, rl_index, dim_number):
    """Carry out the training of the embeddings"""
    # Iterate through each book
    pairs = list()
    for i in range(0, len(log_df)):
        # Iterate through the links in the book
        pairs.append((ac_index[log_df.iloc[i]['task']],
                      rl_index[log_df.iloc[i]['role']]))        
    model = ac_rl_embedding_model(ac_index, rl_index, dim_number)
    #print("ac_index size:", len(ac_index))
    #print("rl_index size:", len(rl_index))
    #print(ac_index)
    #time.sleep(30)
    #print(rl_index)
    model.summary()    
    n_positive = 1024
    gen = generate_batch(pairs, ac_index, rl_index, n_positive, negative_ratio=2)
    #print("Steps per epoch:", len(pairs) // n_positive)
    steps = max(1, len(pairs) // n_positive)    
    # Train
    #model.fit_generator(gen, epochs=100, steps_per_epoch=len(pairs) // n_positive, verbose=2)
    #model.fit(gen, epochs=100, steps_per_epoch=len(pairs) // n_positive, verbose=2)
    model.fit(gen, epochs=100, steps_per_epoch=steps, verbose=2)
    # Extract embeddings
    ac_layer = model.get_layer('activity_embedding')
    rl_layer = model.get_layer('role_embedding')
    ac_weights = ac_layer.get_weights()[0]
    rl_weights = rl_layer.get_weights()[0]
    return ac_weights, rl_weights


def generate_batch(pairs, ac_index, rl_index, n_positive=50,
                   negative_ratio=1.0):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    pairs_set = set(pairs)
    activities = list(ac_index.keys())
    roles = list(rl_index.keys())
    # we cannot pick a larger sample than all elements of a list
    if n_positive > len(pairs):
        n_positive = min(n_positive, len(pairs))        
    # This creates a generator
    while True:
        # randomly choose positive examples
        idx = 0
        for idx, (activity, role) in enumerate(random.sample(pairs,
                                                             n_positive)):
            batch[idx, :] = (activity, role, 1)
        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:
            # random selection
            random_ac = random.randrange(len(activities))
            random_rl = random.randrange(len(roles))

            # Check to make sure this is not a positive example
            if (random_ac, random_rl) not in pairs_set:

                # Add to batch and increment index,  0 due classification task
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'activity': batch[:, 0], 'role': batch[:, 1]}, batch[:, 2]


def ac_rl_embedding_model(ac_index, rl_index, embedding_size):
    """Model to embed activities and roles using the functional API"""
    # Both inputs are 1-dimensional
    activity = Input(name='activity', shape=[1])
    role = Input(name='role', shape=[1])
    # Embedding the activity (shape will be (None, 1, embedding_size))
    activity_embedding = Embedding(name='activity_embedding',
                                   input_dim=len(ac_index),
                                   output_dim=embedding_size)(activity)
    # Embedding the role (shape will be (None, 1, embedding_size))
    role_embedding = Embedding(name='role_embedding',
                               input_dim=len(rl_index),
                               output_dim=embedding_size)(role)
    # Merge the layers with a dot product
    # along the second axis (shape will be (None, 1, 1))
    merged = Dot(name='dot_product',
                 normalize=True, axes=2)([activity_embedding, role_embedding])
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1])(merged)
    # Loss function is mean squared error
    model = Model(inputs=[activity, role], outputs=merged)
    model.compile(optimizer='Adam', loss='mse')

    return model

def reformat_matrix(index, weigths):
    """Reformating of the embedded matrix for exporting.
    Args:
        index: index of activities or roles.
        weigths: matrix of calculated coordinates.
    Returns:
        matrix with indexes.
    """
    matrix = list()
    for i, _ in enumerate(index):
        data = [i, index[i]]
        data.extend(weigths[i])
        matrix.append(data)
    return matrix
