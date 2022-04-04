from math import sqrt
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# By: André Franzén (af223kr)
# Course: 2DV516
# Date: Mars 2022 

# -- 2x2 plot
fig, ([k1plt, k3plt], [k5plt, k7plt]) = plt.subplots(2,2)
fig.suptitle('Plots showing K-NN')

plots = [(k1plt,1), (k3plt,3), (k5plt,5), (k7plt,7)]

# -- (1) Load the data
PATH = "csvFiles/microchips.csv"
chips = pd.read_csv(PATH)

# - premake array instead of reading file to make it quicker, This is only dun once
points = []
for index, row in chips.iterrows():
    points.append((row[0], row[1], row[2]))

x_val = np.array([ (x[0],x[1]) for x in points])

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

def eucDistance(X):
    #From lecture https://github.com/rafaelmessias/2dv516/blob/master/2dv516-python-2-part-1-broadcasting.ipynb
    # But modefied by me
    n = X.shape[0]
    Z = np.empty((n, n))
    for i in range(n):        
        Z[i] = np.sqrt(np.sum((X[i] - X)**2, axis=-1))
    
    #Where Z[1][5] is distance from point 1 to 5
    return Z

stepSize = 0.2
def boundry():
    
    xx, yy = np.meshgrid(np.arange(-1, 1.4, stepSize), np.arange(-1, 1.4, stepSize))
    #A list with all XY cords for entire mesh
    formatedXYList = np.c_[xx.ravel(), yy.ravel()]

    d_matrix = eucDistance(formatedXYList)
    print(d_matrix.shape)
    
#Peredict value function based on KNN
def predictValue(z,k):
    #d_matrix such that d_matrix[10][15] for example, contains the distance between rows 10 and 15 from the original dataset x_val.
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
    xx, yy = np.meshgrid(np.arange(-1, 1.4, stepSize), np.arange(-1, 1.4, stepSize))
    #A list with all XY cords for entire mesh
    formatedXYList = np.c_[xx.ravel(), yy.ravel()]
    cmap_light = ListedColormap(["blue", "red"])
    Z = []
    for x,y in formatedXYList:
        Z.append(predictValue((x,y),k))
    Z = np.array(Z)    

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plot.contourf(xx, yy, Z, alpha=0.7, cmap=cmap_light)

# -- Now compute the errors
# Go to all dots we know and see if they are in correct are area
def checkErrors(k):
    errors = 0
    for x,y,ok in points:
        if predictValue((x,y),k) != ok: 
            errors +=1
    return errors

for plot,k in plots:
    makeboundryPlot(plot, k)
    plot.set_title(f'K = {k}, Training errors = {checkErrors(k)}')

plt.show()
