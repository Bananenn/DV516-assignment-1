
from math import sqrt
from csv import reader
import pandas as pd
import matplotlib.pyplot as plt

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

def predictValue(z,k):
    for index, row in chips.iterrows():
        #d=√((x_2-x_1)²+(y_2-y_1)²)
        dis = sqrt( pow((z[0] - row[0]),2) + pow((z[1] - row[1]),2) )
        currentPoint = ((row[0], row[1], row[2]),dis)
        distanceList.append(currentPoint)

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

for point in zlst:
    for k in klst:
        print(f"{predictValue(point, k)} is the predicted value for {point} with K = {k}")