import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# By: André Franzén (af223kr)
# Course: 2DV516
# Date: Mars 2022 

# (1) Devide the data into a training set (100) and a test set (100)
PATH = "csvFiles/polynomial200.csv"
chips = pd.read_csv(PATH)

# - premake array instead of reading file to make it quicker, This is only dun once
points = []
for index, row in chips.iterrows():
    points.append((row[0], row[1]))

trainingSet = points[:100]
testSet = points[100:]

# (2) Plot training and test set in a 1x2
datafig, ([trainingplt, testplt]) = plt.subplots(1, 2)
datafig.suptitle('Polynomial')
trainingplt.set_title('Training data')
testplt.set_title('Test data')

# -- Setup plots
fig, ([k1plt, k3plt], [k5plt,k7plt], [k9plt,k11plt]) = plt.subplots(3, 2)
fig.suptitle('Polynomial')
trainingplt.set_title('Training data')

#plot both graphs
for x,y in trainingSet:
    trainingplt.plot(x,y,'yx')
    k1plt.plot(x,y,'yx')
    k3plt.plot(x,y,'yx')
    k5plt.plot(x,y,'yx')
    k7plt.plot(x,y,'yx')
    k9plt.plot(x,y,'yx')
    k11plt.plot(x,y,'yx')

for x,y in testSet:
    testplt.plot(x,y,'yx')


def findClosestInX(xval, k, set):
    #This function will return a list of the K closest points in the set from value xval
    pointWithDistance = [] # This will be a list with the point and x distance 
    for point in set:
        pointWithDistance.append((point, abs(point[0]-xval)))

    # -- Sort the list in accending order by Xvalue
        pointWithDistance.sort(key=lambda tup: tup[1])

    # -- Pick the first K from list
        tempList = []
        for point in pointWithDistance[:k]:
            tempList.append(point) #Append Y value
    return tempList

stepSize = 0.05
xRange = np.arange(0, 25, stepSize)
# For each value in xRange take the k datapoint closest and compute Y value 
def plotRegression(plot, k):
    #This function will plot the regression line on plot with k value
    cords = []
    for xval in xRange:
        #Go through every data point find the k closest X values  and grab te Y value
        yOnly = [x[0][1] for x in findClosestInX(xval, k, trainingSet)]
        cords.append((xval , sum(yOnly) / k)) #take avrage Y value

    x_val = [x[0] for x in cords]
    y_val = [x[1] for x in cords]    
    plot.plot(x_val,y_val)
    return cords



# -- (4) Compute and present the MSE test error.
def mseCompute(regLineCords, cordsSet = trainingSet):
    diffInY = []
    for point in cordsSet: #för alla points i set
        diffInY.append(findClosestInX(point[0],1,regLineCords)[0][0][1]-point[1]) #Hitta den närmaste i X led
    res = 0
    for x in diffInY: #Square and sum all values
        res += pow(x, 2)

    res *= 1/len(diffInY) # As formula says 
    return round(res,1)

# -- The lines below do the following.
#Calls on plotRegression to do the ploting then takes the returned array of coorinates and calculates mse and rounds it and updates the title
for plot,k in [(k1plt,1),(k3plt,2),(k5plt,5),(k7plt,7),(k9plt,9),(k11plt,11)]:
    regCords = plotRegression(plot,k)
    plot.set_title(f'K = {k} MSE training error: {mseCompute(regCords)} MSE test error: {mseCompute(regCords,testSet)}')

plt.show()