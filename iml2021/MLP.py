#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score

# Author: Antonio Sutera & Yann Claess
def load_data(data_path):

    FEATURES = range(2, 33)
    N_TIME_SERIES = 3500

    # Create the training and testing samples
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

    for f in FEATURES:
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-2+1)*512] = data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data
    
    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test

# Author: Antonio Sutera & Yann Claess
def write_submission(y, where, submission_name='toy_submission.csv'):

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


# In[2]:


if __name__ == '__main__':
    # Directory containing the data folders
    DATA_PATH = 'data'
    X_train, y_train, X_test = load_data(DATA_PATH)


# In[3]:


# Replace missing values
imputer = KNNImputer(n_neighbors = 5, weights = 'distance', missing_values = -999999.99)
X_train = imputer.fit_transform(X_train)


# In[ ]:


# Feature selection

print("Shape before feature selection: " + str(X_train.shape))

selector = SelectFromModel(estimator = etc).fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

print("Shape after feature selection: " + str(X_train.shape))


# In[5]:


clf = MLPClassifier(hidden_layer_sizes = (500,), random_state = 0)
scores = cross_val_score(clf, X_train, y_train, cv = 10, n_jobs = -1)
score = scores.mean()

print(score)


# In[6]:


clf.fit(X_train, y_train)

y_test = clf.predict(X_test)

write_submission(y_test, 'submissions')


# In[ ]:




