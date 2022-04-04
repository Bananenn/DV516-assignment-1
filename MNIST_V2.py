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


mndata = MNIST('samples/')
images, labels = mndata.load_training()

# All the data
d = np.array(images)
l = np.array(labels)

# seperate training data
numtrain = 10000
DataTrain = d[:numtrain]
lableTrain = l[:numtrain]

#TODO test print for img
# -- The below (How to plot the figures) was learnt from https://benmilanko.com/projects/mnist_with_pca/
fig, ([p1, p2], [p3,p4], [p5,p6]) = plt.subplots(3, 2)
for plot in [p1,p2,p3,p4,p5,p6]:
    pixels = DataTrain[5]
    pixels = pixels.reshape((28,28))
    plot.imshow(pixels, cmap='gray_r')

k = 3
def predictValue(y):
    distanceList = []
    for train, lable in zip(DataTrain, lableTrain):
        train = train.reshape((28,28))
        dis = np.linalg.norm(train-y)
        distanceList.append((lable,dis))

    # -- Sort the list in accending order 
    distanceList.sort(key=lambda tup: tup[1])
        
    # -- Get the top K rows
    # I will for k check and just save all of them in a temp list and then check the most common
    tempList = []
    for item in distanceList[:k]:
        #item[0] to select the point and not distance [2] to select 3rd element that is Ok / not ok
        tempList.append(item[0])

    # -- Get most common from the now k long list
    # - The row below to get the most common value is from https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    return(max(set(tempList), key = tempList.count))

correct = 0
for i in range(100):
    r = random.randint(10000, 50000)
    pred_val = predictValue(d[r].reshape((28,28)))
    print(f"Value is: {l[r]} And the perdicted is {pred_val}") 
    if l[r] == pred_val:
        correct += 1

print(correct)