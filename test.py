from cgi import test
from dis import dis
from math import sqrt
from re import X
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def broadcast1(X):
    n = X.shape[0]
    Z = np.empty((n, n))
    for i in range(n):        
        Z[i] = np.sqrt(np.sum((X[i] - X)**2, axis=-1))
    return Z

    # -- (1) Load the data
PATH = "csvFiles/microchips.csv"
chips = pd.read_csv(PATH)

# - premake array instead of reading file to make it quicker, This is only dun once
points = []
for index, row in chips.iterrows():
    points.append((row[0], row[1], row[2]))

x_val = np.array([ (x[0],x[1]) for x in points])
y_val = np.array([ x[2] for x in points])


smallY_val = y_val[:20]


SearchCords = x_val[:3]
AllPoints = x_val[10:15]


#(1) gör en array all punkter som ska sökas 
searchIndex = SearchCords.shape[0]
allCords = np.vstack([SearchCords,AllPoints])
#print(allCords)

#Make into distance matrix
disMatrix = broadcast1(allCords)

#Take each row until searchIndex
#TODO FIX so K can be larger than 1 currently allways one
for i in range(searchIndex):
    disRow = disMatrix[i].tolist()
    disRow[disRow.index(0)] = 99 #TODO prob not good
    index = disRow.index(min(disRow))
    print("Index: ",index, " Cord: ", allCords[index])
    lab = y_val[index]

    
#lable 


#Now we have lowest distance we need this index



#print(np.append(smallX_val,test))
#lätt till SmallX till testX

#Jag vill ha testXs avstånd till alla andra vektorer, 
#allstå vill jag att dis vec ska blu testXlång
#print(testX)
#print(broadcast1(testX))
#broadcast1(testX)[0][2]
#Jag vill ha distance från Text X till närmaste vektor