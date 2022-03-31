from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Sources:
# https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

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

pointsTrain = points[:100]
pointsTest = points[100:]

x_val = [ (x[0],x[1]) for x in pointsTrain]
y_val = [x[2] for x in pointsTrain] 

pointsTest = [ (x[0],x[1]) for x in pointsTest]

#This creates the class? and sets K to 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit( x_val, y_val ) #Here the trinaind data is input

# - A funcion that takes a point (x,y) and perdicts the outcome
def perdictPoint(toTest):
    toTest = np.reshape(np.array(toTest), (1,-1))
    return knn.predict( toTest )[0]

def perdictMany(toTest):
    return np.reshape(knn.predict( toTest ), (1,-1)) 

stepSize = 0.024 # want 100 steps 2.4/100 = 0.024
xRange = np.arange(-1.0, 1.4, stepSize)
yRange = np.arange(-1.0, 1.4, stepSize)

def improvedPerdictionBoundry(plot):
    allXYCords = []
    for x in xRange:
        for y in yRange:
            allXYCords.append((x,y))

    allXYCords = np.array(allXYCords)
    allResults = np.array(perdictMany(allXYCords))
    cordsWithO = np.append(allXYCords,allResults.T, axis=1)
    
    for x,y,ok in cordsWithO:
        if ok == 0:
            rec = plt.Rectangle((x,y), stepSize, stepSize, fc='blue', alpha=0.5)
            plot.add_patch(rec) #Thease are semi transparant rectangels
        else:
            rec = plt.Rectangle((x,y), stepSize, stepSize, fc='red', alpha=0.5)
            plot.add_patch(rec) #Thease are semi transparant rectangels
    

def makePerdictionBoundry(plot):
    # X and Y ranges from -1 to 1.4 found to be good size
    stepSize = 0.024 # want 100 steps 2.4/100 = 0.024
    xRange = np.arange(-1.0, 1.4, stepSize)
    yRange = np.arange(-1.0, 1.4, stepSize)

    for x in xRange:
        #progress = round(((1+x)/2.4)*100)
        #if (progress % 10) == 0: print(f"Progress: {progress}%") #Simple progress meter will pring every 10% of progress
        for y in yRange:
            if perdictPoint((x,y)) == 0:
                rec = plt.Rectangle((x,y), stepSize, stepSize, fc='blue', alpha=0.5)
                plot.add_patch(rec) #Thease are semi transparant rectangels
            else:
                rec = plt.Rectangle((x,y), stepSize, stepSize, fc='red', alpha=0.5)
                plot.add_patch(rec) #Thease are semi transparant rectangels

#15s
#makePerdictionBoundry(k3plt)

#10s
improvedPerdictionBoundry(k3plt)

k3plt.plot(-1,-1)#TO show full plot for some reason wont show negative otherwise
#makePerdictionBoundry(k3plt)

plt.show()
