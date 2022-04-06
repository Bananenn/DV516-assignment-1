from math import sqrt
import random
from re import T
from mnist import MNIST # This only helps with reading the files weirdidx3-ubyte
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# By: André Franzén (af223kr)
# Course: 2DV516
# Date: Mars 2022 
# Alot of sources are used as this task was complicated.
# They are mentioned in the code
#Additional sources:
# https://benmilanko.com/projects/mnist_with_pca/
# https://www.analyticsvidhya.com/blog/2021/11/pca-on-mnist-dataset/
# https://www.codingninjas.com/codestudio/library/applying-pca-on-mnist-dataset

# -- The below (How to plot the thing) was learnt from https://benmilanko.com/projects/mnist_with_pca/
mndata = MNIST('samples/')
images, labels = mndata.load_training()

t = np.array(images)
l = np.array(labels)

# Lets just use the 1000 first
numtrain = 20000
train = t[:numtrain]
lable = l[:numtrain]

def markOnPlot():
    lable = l[:numtrain]
    lable = np.array(lable)
    # -- Now lets try to PCA!
    # Meaned should help when calculating the covariance matrix (according to guide :) )
    pixels_Meaned = train - np.mean(train, axis=0)
    
    # Calculating the covariance matrix of meaned centered data TODO What does this do? I have no clue
    cov_mat = np.cov(pixels_Meaned, rowvar=False)

    #Calculating the Eigenvalues and Eigenvectors of the covariance matrix TODO också fatta egenvärden och egenvektorer????
    eigeb_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sortedIndex = np.argsort(eigeb_values)[::-1] # Decending order'
    sortedEigenvalue = eigeb_values[sortedIndex]
    sortedEgenvektorer = eigen_vectors[:,sortedIndex]
    eigenVector_subset = sortedEgenvektorer[:,0:2].T
    #eigen_vectors = eigen_vectors[782:]

    newCordinates = np.matmul(eigenVector_subset, pixels_Meaned.T)
    newCordinates = np.vstack( (newCordinates, lable) ).T
    
    for x,y,lable in newCordinates:
        if lable == 0:
            plt.plot(x,y, 'b.')
        elif lable == 1:
            plt.plot(x,y, 'g.')
        elif lable == 2:
            plt.plot(x,y, 'r.')
        elif lable == 3:
            plt.plot(x,y, 'c.')
        elif lable == 4:
            plt.plot(x,y, 'm.')
        elif lable == 5:
            plt.plot(x,y, 'y.')
        elif lable == 6:
            plt.plot(x,y, color='gray',marker='.')
        elif lable == 7:
            plt.plot(x,y, color='peru',marker='.')
        elif lable == 8:
            plt.plot(x,y, color='seagreen',marker='.')
        elif lable == 9:
            plt.plot(x,y, color='hotpink',marker='.')
        
        
    
    return  newCordinates, eigenVector_subset

points, egenVec = markOnPlot()

def getRelCords(input):
    newCordinates = np.matmul(egenVec, input.T).T
    return  newCordinates


newp = []
for i in range(numtrain):
    temp = getRelCords(t[i])
    newp.append((temp[0], temp[1], l[i]))

def predictValue(z,k):
    distanceList = []
    for x,y,ok in newp:
        #d=√((x_2-x_1)²+(y_2-y_1)²)
        dis = sqrt( pow((z[0] - x),2) + pow((z[1] - y),2) )
        distanceList.append(((x, y, ok),dis))

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

def makeboundryPlot(k):
    # X and Y ranges from -1 to 1.4 found to be good size
    # More or less same as excercise 1
    stepSize = 50 
    xRange = np.arange(-1000, 1500, stepSize)
    yRange = np.arange(-1000, 1500, stepSize)

    for x in xRange:
        print(f"Progress: {round(((1000+x)/2500)*100)}%")
        for y in yRange:
            preval = predictValue((x,y),k)    
            if preval == 0:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='blue', alpha=0.5))
            elif preval == 1:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='gold', alpha=0.5))
            elif preval == 2:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='green', alpha=0.5))
            elif preval == 3:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='red', alpha=0.5))
            elif preval == 4:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='orange', alpha=0.5))
            elif preval == 5:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='black', alpha=0.5))
            elif preval == 6:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='yellow', alpha=0.5))
            elif preval == 7:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='brown', alpha=0.5))
            elif preval == 8:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='lime', alpha=0.5))
            elif preval == 9:
                plt.gca().add_patch(plt.Rectangle((x,y), stepSize, stepSize, fc='pink', alpha=0.5))

#makeboundryPlot(3)

fig, ([p1, p2], [p3,p4], [p5,p6]) = plt.subplots(3, 2)
fig.suptitle('perdict values')

plt.show()

"""
for plot in [p1,p2,p3,p4,p5,p6]:
    rand = random.randint(0,5000)
    img = train[rand]
    # Inseration for how to plot the images was taken from.
    # Source: https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot
    pixels = np.array(img)
    pixels = pixels.reshape(28,28)
    print(pixels)
    plot.set_title(f"This is: {predictValue(getRelCords(img),3)}")
    plot.imshow(pixels, cmap='gray_r')

plt.show()


currect = 0
for i in range(50000,50100):
    #Perdicted valye
    p = predictValue(getRelCords(t[i]),3)
    real = l[i]
    if p == real:
        currect +=1

print(currect)

"""