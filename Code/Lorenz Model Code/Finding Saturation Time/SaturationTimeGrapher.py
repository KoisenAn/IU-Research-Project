#python -N pip install -U matplotlib
import matplotlib.pyplot as plt
#python -N pip install -U numpy
import numpy as np
import math
from random import sample

import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the file in the parent directory
parent_dir = os.path.dirname(current_dir)

import sys         
 
# appending the directory of LorenzModel.py 
# in the sys.path list
sys.path.append(parent_dir) 

import LorenzModel

'''
# Graphs Average Error For A Given Perturbation
Range = 2500

plt.xlabel("Time")
plt.ylabel("Average Error")
for dataset in errList:

    plt.plot(perTimeArray[0:Range], dataset[1][0:Range], color = 'red', label = '10⁻⁴ X Perturbation')
    plt.axvline(x = dataset[4], color = 'r', linestyle = 'dashed', label = f"X Saturation Time = {dataset[4]}")

    plt.plot(perTimeArray[0:Range], dataset[2][0:Range], color = 'blue', label = '10⁻⁴  Y Perturbation')
    plt.axvline(x = dataset[5], color = 'blue', linestyle = 'dashed', label = f"Y Saturation Time = {dataset[5]}")

    plt.plot(perTimeArray[0:Range], dataset[3][0:Range], color = 'green', label = '10⁻⁴ Z Perturbation')
    plt.axvline(x = dataset[6], color = 'green', linestyle = 'dashed', label = f"Z Saturation Time = {dataset[6]}")

plt.legend(loc = "upper left")
plt.show()
'''