import numpy as np

array = [np.array([[1, 1, 1, 1], [2, 2, 2, 2],
                  [3, 3, 3, 3], [4, 4, 4, 4]])]

array = np.delete(array, 2, axis = 1)

print(array)
