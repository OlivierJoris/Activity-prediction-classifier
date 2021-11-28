import os
import numpy as np

from matplotlib import pyplot as plt

def computeProportionSubjectMeasures():
    LS = np.loadtxt(os.path.join('../data/LS', 'subject_Id.txt'))
    activity = np.loadtxt(os.path.join('../data/LS', 'activity_Id.txt'))

    uniqueActivity, countActivity = np.unique(activity, return_counts = True)
    uniqueLS, countLS = np.unique(LS, return_counts = True)
    
    print('activity : {}.'.format(np.asarray((uniqueActivity, countActivity))))
    print('LS subject_id : {}.'.format(np.asarray((uniqueLS, countLS))))
    
if __name__ == '__main__':
    computeProportionSubjectMeasures()

