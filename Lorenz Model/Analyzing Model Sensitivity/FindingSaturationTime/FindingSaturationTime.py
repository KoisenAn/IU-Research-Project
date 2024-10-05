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
 
# appending the directory of mod.py 
# in the sys.path list
sys.path.append(parent_dir) 

import LorenzModel

#
# Starting Code
#

# Time steps/Increment values
dt = 0.01
T = 1000
N = round(T/dt) + 1

# Base Model Data Set
baseModelX = np.zeros(N)
baseModelY = np.zeros(N)
baseModelZ = np.zeros(N)
baseTimeArray = np.zeros(N)

# Base Initial Values
baseModelX[0] = 0.1
baseModelY[0] = 0.1
baseModelZ[0] = 0.1

LorenzModel.GenerateEulerScheme(baseModelX, baseModelY, baseModelZ, baseTimeArray, N)

#
# Testing Perturbations
#

perturbationList = []

'''
#Adding Initial 0.1 Perturbation To X 

perturbationList.append(("x", 0.1))
'''

'''
#Adding Initial 0.1 Perturbation To Y 

perturbationList.append(("y", 0.1))
'''

'''
#Adding Initial 0.1 Perturbation To Z 

perturbationList.append(("z", 0.1))
'''

'''
#Adding Initial 0.01 Perturbation To X 

perturbationList.append(("x", 0.01))
'''

'''
#Adding Initial 0.01 Perturbation To Y 

perturbationList.append(("y", 0.01))
'''

'''
#Adding Initial 0.01 Perturbation To Z 

perturbationList.append(("z", 0.01))
'''

'''
#Adding Initial 0.001 Perturbation To X 

perturbationList.append(("x", 0.001))
'''

'''
#Adding Initial 0.001 Perturbation To Y 

perturbationList.append(("y", 0.001))
'''

'''
#Adding Initial 0.001 Perturbation To Z 

perturbationList.append(("z", 0.001))
'''

#Adding Initial 0.0001 Perturbation To X 

perturbationList.append(("x", 0.0001))

'''
#Adding Initial 0.00001 Perturbation To Y 

perturbationList.append(("y", 0.0001))
'''

'''
#Adding Initial 0.00001 Perturbation To Z 

perturbationList.append(("z", 0.0001))
'''

#
# Finding Saturation Time
#

trials = 1000
lowerTimeBound = 1
upperTimeBound = 10000

errList = []

for perturbation in perturbationList:

    trialValues = []

    # This is used to generate the list of indexes that perturbations will be added to. 
    # We do this to introduce multiple perturbations of the same magnitude at different initial times, allowing us to treat them as small variations to introduce flexibility in our saturation time calculation.
    # For each trial, we will only consider the values after the perturbation is added to determine the error.
    randomIndexList = sample(range(lowerTimeBound, upperTimeBound), trials)

    # Status Update
    i = 0

    for index in randomIndexList:
        
        # Status Update
        i += 1
        print(f"{i}/{trials}")

        # Perturbed Model Data Set
        perModelX = np.zeros(N)
        perModelY = np.zeros(N)
        perModelZ = np.zeros(N)
        perTimeArray = np.zeros(N)

        # Perturbed Model Initial Conditions
        perModelX[0] = 0.1
        perModelY[0] = 0.1
        perModelZ[0] = 0.1

        #Generating perturbed data set values

        LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, index, perturbation[1], variable = perturbation[0])
        
        #Saving values
        trialValues.append((perModelX, perModelY, perModelZ))

    #
    # Finding the average values of all the trials
    #

    # Readjusts max range because of random index selection. 
    M = N - max(randomIndexList)

    # Data sets for the sums of all the trials
    sumXtrials = np.zeros(M)
    sumYtrials = np.zeros(M)
    sumZtrials = np.zeros(M)

    for dataset, index in zip(trialValues, randomIndexList):
        #Computes general error for the each entire data set. NOTE: Until the perturbation is added, there is zero error. So we don't consider the values before the perturbation is added.
        errX = abs(dataset[0][:] - baseModelX[:])
        errY = abs(dataset[1][:] - baseModelY[:])
        errZ = abs(dataset[2][:] - baseModelZ[:])
        
        # Only registers the values after the perturbation is added
        sumXtrials[:] += errX[index:index + M]
        sumYtrials[:] += errY[index:index + M]
        sumZtrials[:] += errZ[index:index + M]

    # Divides the data sets by the number of trials to find the average
    averageXtrials = sumXtrials/len(trialValues)
    averageYtrials = sumYtrials/len(trialValues)
    averageZtrials = sumZtrials/len(trialValues)
    
    error = 0.01
    Range = 1000
    overError = 1

    SaturationTimeX = round(LorenzModel.findSaturationTime(averageXtrials, perTimeArray[:M], error = error, range = Range, overError = overError), 2)
    SaturationTimeY = round(LorenzModel.findSaturationTime(averageYtrials, perTimeArray[:M], error = error, range = Range, overError = overError), 2)
    SaturationTimeZ = round(LorenzModel.findSaturationTime(averageZtrials, perTimeArray[:M], error = error, range = Range, overError = overError), 2)

    print(SaturationTimeX)
    print(SaturationTimeY)
    print(SaturationTimeZ)

    # Saving Data To Graph
    errList.append((perturbationList, averageXtrials, averageYtrials, averageZtrials, round(SaturationTimeX, 2), round(SaturationTimeY, 2), round(SaturationTimeZ, 2)))

    '''
    # Saving Data To File
    f = open(f"FindingSaturationTimeData_{perturbation[0]}_{perturbation[1]}.txt", "w")

    f.write(f"Perturbation Added To: {perturbation[0]}")
    f.write("\n")
    f.write(f"Perturbation: {perturbation[1]}")
    f.write("\n")
    f.write(f"Saturation Time X: {SaturationTimeX}")
    f.write("\n")
    f.write(f"Saturation Time Y: {SaturationTimeY}")
    f.write("\n")
    f.write(f"Saturation Time Z: {SaturationTimeZ}")
    f.write("\n")

    for i in range(M):
        f.write(f"{averageXtrials[i]} {averageYtrials[i]} {averageZtrials[i]}")
        f.write("\n")

    f.close()
    '''


#
# Graphs
#


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