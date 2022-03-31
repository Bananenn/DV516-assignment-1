from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# By: André Franzén (af223kr)
# Course: 2DV516
# Date: Mars 2022 

# -- 2x2 plot
fig, ([k1plt, k3plt], [k5plt, k7plt]) = plt.subplots(2,2)
fig.suptitle('Plots showing K-NN')


# -- (1) Load the data
PATH = "microchips.csv"
chips = pd.read_csv(PATH)

# - premake array instead of reading file to make it quicker, This is only dun once
points = []
for index, row in chips.iterrows():
    points.append((row[0], row[1], row[2]))

# -- (2) initialize the value of K
klst = [1,3,5,7]

# -- (3) Assume we want to classify a datapoint Z
zlst = [(-0.3, 1.0), (-0.5, -0.1), (0.6, 0.0)]

# - Print all points to make the plot make sense
for x,y,ok in points:
    if ok == 0:
        k1plt.plot(x, y, 'r.')
        k3plt.plot(x, y, 'r.')
        k5plt.plot(x, y, 'r.')
        k7plt.plot(x, y, 'r.')
    else:
        k1plt.plot(x, y, 'b.')
        k3plt.plot(x, y, 'b.')
        k5plt.plot(x, y, 'b.')
        k7plt.plot(x, y, 'b.')
    

#Peredict value function based on KNN
def predictValue(z,k):
    distanceList = []
    for x,y,ok in points:
        #d=√((x_2-x_1)²+(y_2-y_1)²)
        dis = sqrt( pow((z[0] - x),2) + pow((z[1] - y),2) )
        distanceList.append( ((x, y, ok),dis) )

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

def makeboundryPlot(plot,k):
    # X and Y ranges from -1 to 1.4 found to be good size
    stepSize = 0.024 # want 100 steps 2.4/100 = 0.024
    xRange = np.arange(-1.0, 1.4, stepSize)
    yRange = np.arange(-1.0, 1.4, stepSize)

    for x in xRange:
        progress = round(((1+x)/2.4)*100)
        if (progress % 10) == 0: print(f"Progress: {progress}%") #Simple progress meter will pring every 10% of progress
        for y in yRange:
            if predictValue((x-stepSize/2,y-stepSize/2),k) == 0:
                plot.add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='red', alpha=0.5)) #Thease are semi transparant rectangels
            else:
                plot.add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='blue', alpha=0.5))

makeboundryPlot(k1plt,1)
makeboundryPlot(k3plt,3)
makeboundryPlot(k5plt,5)
makeboundryPlot(k7plt,5)

# -- Now compute the errors
# Go to all dots we know and see if they are in correct are area
def checkErrors(k):
    errors = 0
    for x,y,ok in points:
        if predictValue((x,y),k) != ok: 
            errors +=1
    return errors

k1plt.set_title(f'K = 1, Training errors = {checkErrors(1)}')
k3plt.set_title(f'K = 3, Training errors = {checkErrors(3)}')
k5plt.set_title(f'K = 5, Training errors = {checkErrors(5)}')
k7plt.set_title(f'K = 5, Training errors = {checkErrors(7)}')

plt.show()