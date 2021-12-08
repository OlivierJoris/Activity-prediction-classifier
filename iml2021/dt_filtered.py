#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Maxime Goffart and Olivier Joris

import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer

class DTFiltered:
    """
    Classifier that uses a decision tree with filtered data.
    """

    def __init__(self, min_sample_split):
        """
        Argument:
        ---------
        - `min_sample_split`: min_sample_split for the tree.
        """
        self.min_sample_split = min_sample_split
    
    def load_data(self, data_path):
        """
        Load the data for the classifer.
        Modified from the method given with the assignment. Authors: Antonio Sutera & Yann Claess.

        Argument:
        ---------
        - `data_path`: Path to the data folder.
        """

        FEATURES = range(2, 33)
        N_TIME_SERIES = 3500

        print("Loading data...")

        # Create the training and testing samples
        LS_path = os.path.join(data_path, 'LS')
        TS_path = os.path.join(data_path, 'TS')
        X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

        for f in FEATURES:
            print("Loading feature {}...".format(f))
            data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
            X_train[:, (f-2)*512:(f-2+1)*512] = data
            data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
            X_test[:, (f-2)*512:(f-2+1)*512] = data
        
        y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

        print('X_train size: {}.'.format(X_train.shape))
        print('y_train size: {}.'.format(y_train.shape))
        print('X_test size: {}.'.format(X_test.shape))

        # Replace missing values
        print("Replace missing values...")
        imputer = KNNImputer(n_neighbors = 5, weights = 'distance', missing_values = -999999.99)
        X_train = imputer.fit_transform(X_train)

        # Features selection
        print("Features selection...")
        etc = ExtraTreesClassifier(n_estimators = 1000)
        
        print("X_train shape before feature selection: " + str(X_train.shape))
        
        print("SelectFromModel...")
        selector = SelectFromModel(estimator = etc).fit(X_train, y_train)
        print("Transform X_train...")
        X_train = selector.transform(X_train)
        print("Transform X_test...")
        X_test = selector.transform(X_test)
        
        print("X_train shape after feature selection: " + str(X_train.shape))
        print("y_train shape after feature selection: " + str(y_train.shape))

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    def fit(self):
        """
        Fit the classifier.
        """

        self.model = DecisionTreeClassifier(min_samples_split=self.min_sample_split)
        self.model = self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Predict the class labels.

        Return:
        -------
        Return the predictions as a numpy ndarray.
        """

        predictions = np.zeros(3500, dtype=int)
        predictions = self.model.predict(self.X_test)

        return predictions


def write_submission(y, where, submission_name='toy_submission.csv'):
    """
    Method given with the assignment. Authors: Antonio Sutera & Yann Claess.

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
    tree = DTFiltered(min_sample_split=2)

    tree.load_data(DATA_PATH)

    print("Fitting...")
    tree.fit()

    print("Predicting ...")
    predictions = np.zeros(3500, dtype=int)
    predictions = tree.predict()

    write_submission(predictions, 'submissions', submission_name='dt_filtered.csv')
