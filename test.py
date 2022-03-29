import numpy as np

point = np.array([-0.3, 1.0])
disToarr = np.array([(-0.5, -0.1), (0.6, 0.0)])
minusarr = disToarr - point
minusarr = np.power(minusarr,2)
print(minusarr)
#disToarr = np.array

