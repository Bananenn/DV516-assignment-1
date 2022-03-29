import math
import numpy as np

import time

#d=√((x_2-x_1)²+(y_2-y_1)²)
point = np.array([-0.3, 1.0])
disToarr = np.array([(-0.5, -0.1), (0.6, 0.0)])

start_time = time.time()
minusarr = disToarr - point
#d=√((x)²+(y)²)
minusarr = np.power(minusarr,2)
#d=√((x)+(y))
sumarr = np.sum(minusarr,axis=1)
#d=√((sum)
distance = np.sqrt(sumarr)
print("Matrix time" + time.time() - start_time)
#print(distance)

# My old way below
dis = math.sqrt( pow((point[0] - disToarr[0][0]),2) + pow((point[1] - disToarr[0][1]),2) )
print(dis)
dis = math.sqrt( pow((point[0] - disToarr[1][0]),2) + pow((point[1] - disToarr[1][1]),2) )
print(dis)

#disToarr = np.array

