import os
import numpy as np


def computeSizeOfDifferentSubjects(subjectNumber):
    LS = np.loadtxt(os.path.join('../data/LS', 'subject_Id.txt'))
    TS = np.loadtxt(os.path.join('../data/TS', 'subject_Id.txt'))

    uniqueLS, countLS = np.unique(LS, return_counts = True)
    uniqueTS, countTS = np.unique(TS, return_counts = True)

    print('LS : {}.'.format(np.asarray((uniqueLS, countLS))))
    print('TS : {}.'.format(np.asarray((uniqueTS, countTS))))

if __name__ == '__main__':
    subjectNumber = 3500
    computeSizeOfDifferentSubjects(subjectNumber)

