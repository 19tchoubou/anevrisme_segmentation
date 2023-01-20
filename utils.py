import os
import numpy as np
from tqdm import tqdm
import h5py
from patchify import patchify # helps cropping images to create more training subimages
import segmentation_models_3D as sm # pretrained models
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def load_data_from(path_folder, crop=None, center_cube_only=False):

    # get file names
    file_names = os.listdir(path_folder)
    N = len(file_names)
    print(f'{N} samples in dataset.')
    print(file_names)

    # open all .h5 files, split inputs and target masks, store all in np.arrays
    raw_data = []
    labels = []
    names = []

    for file_name in tqdm(file_names):
        f = h5py.File(f'{path_folder}/{file_name}', 'r')

        X, Y = np.array(f['raw']), np.array(f['label'])

        if crop is None:
            raw_data.append(X)
            labels.append(Y)
            names.append(file_name)

        else:
            if center_cube_only: # only keep the center cube (over 9 candidates)
                X = X[:,crop:2*crop,crop:2*crop]
                Y = Y[:,crop:2*crop,crop:2*crop]

                raw_data.append(X)
                labels.append(Y)
                names.append(file_name)

            else: # keep all cubes = more data
                X_patches = patchify(X, (64, 64, 64), step=64)  # Step=64 for 64 patches means no overlap
                X_patches_resh = np.reshape(X_patches, (-1, X_patches.shape[3], X_patches.shape[4], X_patches.shape[5]))
                Y_patches = patchify(Y, (64, 64, 64), step=64)  # Step=64 for 64 patches means no overlap
                Y_patches_resh = np.reshape(Y_patches, (-1, Y_patches.shape[3], Y_patches.shape[4], Y_patches.shape[5]))
                raw_data.append(X_patches_resh)
                labels.append(Y_patches_resh)
                names.append(file_name)

    # convert to arrays for patchify
    raw_data = np.array(raw_data)
    labels = np.array(labels)

    if (crop is not None) and (not center_cube_only): # only keep the center cube (over 9 candidates)
        raw_data = np.reshape(raw_data, (-1, raw_data.shape[2], raw_data.shape[3], raw_data.shape[4]))
        labels = np.reshape(labels, (-1, labels.shape[2], labels.shape[3], labels.shape[4]))

    return raw_data, labels, names


def analytics(y_test, y_pred01, threshold=0.5):
    print(f'------ AFTER THRESHOLDING AT {threshold} ------')
    print('> sm.metrics.IOUScore :', sm.metrics.IOUScore()(y_test, y_pred01))

    # precision_recall_fscore_support report
    precision, recall, fscore, support = precision_recall_fscore_support(y_test.flatten(), 
                                                                      y_pred01.flatten()) 
    print('> Precision :', precision[1])
    print('> Recall :', recall[1])
    print('> Fscore :', fscore[1])