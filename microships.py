
from math import sqrt
from csv import reader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -- (1) Load the data
PATH = "microchips.csv"
chips = pd.read_csv(PATH)

# -- (2) initialize the value of K
klst = [1,3,5,7]

# -- (3) Assume we want to classify a datapoint Z
zlst = [(-0.3, 1.0), (-0.5, -0.1), (0.6, 0.0)]

# -- (4) find the K nearest granne to Z
# my list will look something like [((x,y,ok),distance),((x,y,ok),distance)]
distanceList = []

#TODO just for resting
for index, row in chips.iterrows():
    if row[2] == 0:
        plt.plot(row[0], row[1], 'r*')
    else:
        plt.plot(row[0], row[1], 'bx')
    
#TODO also temp plot Z
plt.plot(zlst[0][0],zlst[0][1], 'yo')
plt.plot(zlst[1][0],zlst[1][1], 'go')
plt.plot(zlst[2][0],zlst[2][1], 'co')

#Peredict value function based on KNN, TODO this needs to be more efficient
def predictValue(z,k):
    distanceList = []
    for index, row in chips.iterrows():
        #d=√((x_2-x_1)²+(y_2-y_1)²)
        dis = sqrt( pow((z[0] - row[0]),2) + pow((z[1] - row[1]),2) )
        currentPoint = ((row[0], row[1], row[2]),dis)
        distanceList.append(currentPoint)

        #[x1, x2] - 

    # -- Sort the list in accending order 
    distanceList.sort(key=lambda tup: tup[1])

    # -- Get the top K rows
    # I will for k check and just save all of them in a temp list and then check the most common
    tempList = []
    for item in distanceList[:k]:
        #item[0] to select the point and not distance [2] to select 3rd element that is Ok / not ok
        tempList.append(item[0][2])

    # -- Get most common from the now k long list
    # - The row below to get the most common value is from https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    return(max(set(tempList), key = tempList.count))

# -- Task 1.2 Perdicton of predefined points
def perdictForPredefPoints():
    for point in zlst:
        print(f"Point: {point}")
        for k in klst:
            print(f"{predictValue(point, k)} is the predicted value with K = {k}")

#perdictForPredefPoints()

# Devide X and Y into smaller parts (will try 50)
#For each part check if it should be 1 or 0 color cordinate

# X and Y ranges from -1 to 1
stepSize = 0.04
xRange = np.arange(-1.0, 1.4, stepSize)
yRange = np.arange(-1.0, 1.4, stepSize)

for x in xRange:
    print(f"Progress: {round(((1+x)/2.4)*100)}%")
    for y in yRange:
        if predictValue((x-stepSize/2,y-stepSize/2),3) == 0:
            plt.gca().add_patch(plt.Rectangle((x-stepSize/2,y-stepSize/2), stepSize, stepSize, fc='red', alpha=0.5))
            
        else:
            plt.gca().add_patch(plt.Rectangle((x-stepSize/2,y-stepSize/2), stepSize, stepSize, fc='blue', alpha=0.5))

plt.show()