import numpy as np
import os
import h5py # to open the .h5 files
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Store all util functions
## Data Generator


def instantiate_generator(PATH_DATASET='./challenge_dataset/', preprocess_function=lambda x : x, batch_size=16, train_size=None, test_size=None):
    """
    Instantiate 2 generators of the data using the specified preprocessing function
    """
    data_exists = os.path.exists(PATH_DATASET)

    if data_exists:
        print(f"Dataset found on device at : '{PATH_DATASET}.'") 
    else:
        raise FileNotFoundError(f"Data folder not found at '{PATH_DATASET}'")

    # get file names in the folder
    file_names = os.listdir(PATH_DATASET)
    N = len(file_names)

    # Raw Data and labels

    raw_data = []
    labels = []
    names = []

    for file_name in file_names:
        f = h5py.File(f'{PATH_DATASET}/{file_name}', 'r')

        X, Y = np.array(f['raw']), np.array(f['label'])

        raw_data.append(np.copy(X))
        labels.append(np.copy(Y))
        names.append(file_name)

    sample_shape = raw_data[0].shape
    label_shape = labels[0].shape
    assert len(labels) == N and len(names) == N, "Inconsistent lengths"

    raw_train, raw_test, label_train, label_test = train_test_split(raw_data, labels, train_size=train_size, test_size=test_size)

    def gen_func1():
        yield from zip(raw_train, label_train)
    def gen_func2():
        yield from zip(raw_test, label_test)
    def map_func(x, y):
        # if need preprocess data
        # write some logic here

        x = preprocess_function(x)
        return x, y
    
    
    train_data_pipeline = tf.data.Dataset.from_generator(gen_func1,
                                                        output_signature=(tf.TensorSpec(shape=sample_shape,dtype=tf.float32),
                                                                        tf.TensorSpec(shape=sample_shape,dtype=tf.uint8)))\
                                        .map(map_func)\
                                        .batch(batch_size)
    val_data_pipeline = tf.data.Dataset.from_generator(gen_func2,
                                                    output_signature=(tf.TensorSpec(shape=sample_shape,dtype=tf.float32),
                                                                    tf.TensorSpec(shape=sample_shape,dtype=tf.uint8)))\
                                    .map(map_func)\
                                    .batch(batch_size)
    
    return train_data_pipeline, val_data_pipeline



