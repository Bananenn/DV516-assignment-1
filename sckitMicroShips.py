from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Sources:
# https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

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

x_val = [ (x[0],x[1]) for x in points]
y_val = [x[2] for x in points] 

#This creates the class? and sets K to 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit( x_val, y_val ) #Here the trinaind data is input

stepSize = 0.024 # want 100 steps 2.4/100 = 0.024

# - A funcion that takes a point (x,y) and perdicts the outcome
def perdictPoint(toTest, k):
    knn.n_neighbors = k
    toTest = np.reshape(np.array(toTest), (1,-1))
    return knn.predict( toTest )[0]

def plotOrgData(plot):
    for cords,ok in zip(x_val, y_val):
        if ok == 1:
            plot.plot(cords[0],cords[1],'r.')
        else:
            plot.plot(cords[0],cords[1],'b.')

def makePerdictionBoundryMesh(plot, k):
    knn.n_neighbors = k
    #Insperation from https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    # Create color maps
    cmap_light = ListedColormap(["blue", "red"])

    # Plot the decision boundary. For that, we will assign a color to each
    xx, yy = np.meshgrid(np.arange(-1, 1.4, stepSize), np.arange(-1, 1.4, stepSize))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plot.contourf(xx, yy, Z, alpha=0.7, cmap=cmap_light)

def checkTrainingError(k):
    errors = 0
    for x,y,ok in points:
        if perdictPoint((x,y),k) != ok: 
            errors +=1
    return errors

#Mesh print perdicton
for plot,k in plots:
    plotOrgData(plot)
    makePerdictionBoundryMesh(plot,k)
    plot.set_title(f'K = {k} Training error: {checkTrainingError(k)}')
    
plt.show()
