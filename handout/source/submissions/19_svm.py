#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Maxime Goffart and Olivier Joris

import os
import numpy as np
import random

from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from scipy import stats
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score

def load_data(data_path):
    """
    Load the data for the classifer.
    Modified from the method given with the assignment. Authors: Antonio Sutera & Yann Claes.

    Argument:
    ---------
    - `data_path`: Path to the data folder.
    """

    FEATURES = range(2, 33)
    N_TIME_SERIES = 3500

    # Create the training and testing samples
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

    for f in FEATURES:
        print("Loading {}".format(f))
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-2+1)*512] = data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data
    
    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test

def write_submission(y, where, submission_name='toy_submission.csv'):
    """
    Method given with the assignment. Authors: Antonio Sutera & Yann Claes.

    Arguments:
    ----------
    - `y`: Predictions to write.
    - `where`: Path to the file in which to write.
    - `submission_name`: Name of the file.
    """

    os.makedirs(where, exist_ok=True)

    SUBMISSION_PATH = os.path.join(where, submission_name)
    if os.path.exists(SUBMISSION_PATH):
        os.remove(SUBMISSION_PATH)

    y = y.astype(int)
    outputs = np.unique(y)

    # Verify conditions on the predictions
    if np.max(outputs) > 14:
        raise ValueError('Class {} does not exist.'.format(np.max(outputs)))
    if np.min(outputs) < 1:
        raise ValueError('Class {} does not exist.'.format(np.min(outputs)))
    
    # Write submission file
    with open(SUBMISSION_PATH, 'a') as file:
        n_samples = len(y)
        if n_samples != 3500:
            raise ValueError('Check the number of predicted values.')

        file.write('Id,Prediction\n')

        for n, i in enumerate(y):
            file.write('{},{}\n'.format(n+1, int(i)))

    print('Submission {} saved in {}.'.format(submission_name, SUBMISSION_PATH))

def feature_extraction(X_train, X_test, data_path):
    """
    Feature extraction.

    Arguments:
    ----------
    - `X_train`: Inputs of LS.
    - `X_test`: Inputs of TS.
    - `data_path`: Path to the data folder.
    
    Return:
    -------
    Inputs of LS and TS after feature extraction.
    """

    FEATURES = range(2, 33)
    THREE_DIM_FEATURES = [4, 7, 10, 14, 17, 20, 24, 27, 30]
    ONE_DIM_FEATURES = [2, 3, 13, 23]
    N_TIME_SERIES = 3500
    N_ATTRIB = 16
    
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    
    # + 1 for the subject id
    new_X_train = np.zeros((N_TIME_SERIES, N_ATTRIB * 31 + 1))
    new_X_test = np.zeros((N_TIME_SERIES, N_ATTRIB * 31 + 1))
    
    LS_subject_id = np.loadtxt(os.path.join(LS_path, 'subject_Id.txt'))
    TS_subject_id = np.loadtxt(os.path.join(TS_path, 'subject_Id.txt'))
    
    # subject id
    new_X_train[:, 0] = LS_subject_id
    new_X_test[:, 0] = TS_subject_id
        
    for i in range(N_TIME_SERIES):
        index = 0
        for f in FEATURES:
            index += 1
            # mean
            new_X_train[i][index] = np.mean(X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.mean(X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # stddev
            new_X_train[i][index] = np.std(X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.std(X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # median
            new_X_train[i][index] = np.median(X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.median(X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # min
            new_X_train[i][index] = np.min(X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.min(X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # max
            new_X_train[i][index] = np.max(X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.max(X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # index of argmax
            new_X_train[i][index] = np.argmax(np.abs(X_train[i][(f-2)*512:(f-2+1)*512]))
            new_X_test[i][index] = np.argmax(np.abs(X_test[i][(f-2)*512:(f-2+1)*512]))

            index += 1
            # index of argmin 
            new_X_train[i][index] = np.argmin(np.abs(X_train[i][(f-2)*512:(f-2+1)*512]))
            new_X_test[i][index] = np.argmin(np.abs(X_test[i][(f-2)*512:(f-2+1)*512]))

            index += 1
            # diff between indexes
            new_X_train[i][index] = abs(np.argmax(np.abs(X_train[i][(f-2)*512:(f-2+1)*512])) - np.argmin(np.abs(X_train[i][(f-2)*512:(f-2+1)*512])))
            new_X_test[i][index] = abs(np.argmax(np.abs(X_test[i][(f-2)*512:(f-2+1)*512])) - np.argmin(np.abs(X_test[i][(f-2)*512:(f-2+1)*512])))

            index += 1
            # median absolute deviation
            new_X_train[i][index] = np.apply_along_axis(lambda x: np.median(np.absolute(x - np.median(x))), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: np.median(np.absolute(x - np.median(x))), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # range
            new_X_train[i][index] = np.max(X_train[i][(f-2)*512:(f-2+1)*512]) - np.min(X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.max(X_test[i][(f-2)*512:(f-2+1)*512]) - np.min(X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # interquartile range
            new_X_train[i][index] = np.apply_along_axis(lambda x: np.percentile(x, 75) - np.percentile(x, 25), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: np.percentile(x, 75) - np.percentile(x, 25), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # positive values
            new_X_train[i][index] = np.apply_along_axis(lambda x: np.sum(x > 0), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: np.sum(x > 0), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # negative values
            new_X_train[i][index] = np.apply_along_axis(lambda x: np.sum(x < 0), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: np.sum(x < 0), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # values above mean
            new_X_train[i][index] = np.apply_along_axis(lambda x: np.sum(x > np.mean(x)), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: np.sum(x > np.mean(x)), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # nb of peaks
            new_X_train[i][index] = np.apply_along_axis(lambda x: len(find_peaks(x)[0]), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: len(find_peaks(x)[0]), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # skewness
            new_X_train[i][index] = np.apply_along_axis(lambda x: stats.skew(x), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: stats.skew(x), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # kurtosis
            new_X_train[i][index] = np.apply_along_axis(lambda x: stats.kurtosis(x), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: stats.kurtosis(x), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # mean absolute deviation
            new_X_train[i][index] = np.apply_along_axis(lambda x: np.mean(np.absolute(x - np.mean(x))), 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: np.mean(np.absolute(x - np.mean(x))), 0, X_test[i][(f-2)*512:(f-2+1)*512])

            index += 1
            # energy 
            new_X_train[i][index] = np.apply_along_axis(lambda x:  np.sum(x**2)/512, 0, X_train[i][(f-2)*512:(f-2+1)*512])
            new_X_test[i][index] = np.apply_along_axis(lambda x: np.sum(x**2)/512, 0, X_test[i][(f-2)*512:(f-2+1)*512])

    return new_X_train, new_X_test

if __name__ == '__main__':
    # Directory containing the data folders
    DATA_PATH = 'data'
    init_X_train, y_train, init_X_test = load_data(DATA_PATH)

    # Replace missing values
    imputer = KNNImputer(n_neighbors = 5, weights = 'distance', missing_values = -999999.99)
    init_X_train = imputer.fit_transform(init_X_train)

    # Feature extraction
    X_train, X_test = feature_extraction(init_X_train, init_X_test, DATA_PATH)

    clf = SVC(probability=True, gamma='auto', random_state = 0).fit(X_train, y_train)
    y_test = clf.predict(X_test)
    write_submission(y_test, 'submissions', submission_name='19_SVM_features_extraction.csv')
