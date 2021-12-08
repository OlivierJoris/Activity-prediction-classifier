#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Maxime Goffart and Olivier Joris

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KnnSplitted:
    """
    Classifier that uses one KNN per feature.
    """

    def __init__(self, n_neighbors):
        """
        Argument:
        ---------
        - `n_neighbors`: number of neighbors used in the KNN models.
        """
        self.n_neighbors = n_neighbors
    
    def load_data(self, data_path):
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
        X_train = [np.zeros((N_TIME_SERIES, 512)) for i in range(2, 33)]
        X_test = [np.zeros((N_TIME_SERIES, 512)) for i in range(2, 33)]

        for f in FEATURES:
            data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
            X_train[f-2] = data
            data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
            X_test[f-2] = data

        y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

        print('X_train len: {}.'.format(len(X_train)))
        print('y_train len: {}.'.format(len(y_train)))
        print('X_test len: {}.'.format(len(X_test)))

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
    
    def fit(self):
        """
        Fit the classifier.
        """

        self.models = []

        for i in range(2, 33):
            model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            model.fit(self.X_train[i-2], self.y_train)
            self.models.append(model)

    def predict(self):
        """
        Predict the class labels.

        Return:
        -------
        Return the predictions as a numpy ndarray.
        """

        predictions = np.zeros((31, 3500), dtype=int)

        for i in range(2, 33):
            pred = np.zeros(3500)
            pred = self.models[i-2].predict(self.X_test[i-2])
            predictions[i-2] = pred

        predictedClasses = np.zeros(3500, dtype=int)
        for i in range (3500):
            predictedClasses[i] = np.argmax(np.bincount(predictions[:, i]))

        return predictedClasses


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

if __name__ == '__main__':

    # Directory containing the data folders
    DATA_PATH = 'data'

    clf = KnnSplitted(n_neighbors=1)
    clf.load_data(DATA_PATH)
    clf.fit()

    predictions = np.zeros(3500, dtype=int)
    predictions = clf.predict()

    write_submission(predictions, 'submissions', submission_name='knn_splitted_1.csv')
