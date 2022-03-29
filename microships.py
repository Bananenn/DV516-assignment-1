
from csv import reader
from re import X
from telnetlib import NOP
from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt

# -- (1) Load the data
PATH = "microchips.csv"
chips = pd.read_csv(PATH)

# -- (2) initialize the value of K
k = [1,3,5,7]

# -- (3) Assume we want to classify a datapoint Z
z = [(-0.3, 1.0), (-0.5, -0.1), (0.6, 0.0)]

# -- (4) find the K nearest granne to Z
# my list will look something like [((x,y,ok),distance),((x,y,ok),distance)]

#TODO just for resting
for index, row in chips.iterrows():
    if row[2] == 0:
        plt.plot(row[0], row[1], 'r*')
    else:
        plt.plot(row[0], row[1], 'bx')
    
    print(f"{row[0]} - {row[1]} - {row[2]}")

#plt.show()
