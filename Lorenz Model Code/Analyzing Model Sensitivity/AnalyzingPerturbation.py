#python -m pip install -U matplotlib
import matplotlib.pyplot as plt
#python -m pip install -U numpy
import numpy as np
import math

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

# Perturbed Model Data Set
perModelX = np.zeros(N)
perModelY = np.zeros(N)
perModelZ = np.zeros(N)
perTimeArray = np.zeros(N)

# Base Initial Values
baseModelX[0] = 0.1
baseModelY[0] = 0.1
baseModelZ[0] = 0.1

# Perturbed Model Initial Conditions
perModelX[0] = 0.1
perModelY[0] = 0.1
perModelZ[0] = 0.1

LorenzModel.GenerateEulerScheme(baseModelX, baseModelY, baseModelZ, baseTimeArray, N)

#
# Testing Perturbations
#

'''
#Investigating Initial 0.1 Perturbation To X 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.1, "x")
'''

'''
#Investigating Initial 0.1 Perturbation To Y 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.1, "y")
'''

'''
#Investigating Initial 0.1 Perturbation To Z 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.01, "z")
'''

'''
#Investigating Initial 0.01 Perturbation To X 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.01, "x")
'''

'''
#Investigating Initial 0.01 Perturbation To Y 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.01, "y")
'''

'''
#Investigating Initial 0.01 Perturbation To Z 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.01, "z")
'''

'''
#Investigating Initial 0.001 Perturbation To X 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.001, "x")
'''

'''
#Investigating Initial 0.001 Perturbation To Y 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.001, "y")
'''

'''
#Investigating Initial 0.001 Perturbation To Z 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.001, "z")
'''

#Investigating Initial 0.0001 Perturbation To X 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.0001, "x")

'''
#Investigating Initial 0.00001 Perturbation To Y 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.0001, "y")
'''

'''
#Investigating Initial 0.00001 Perturbation To Z 

LorenzModel.AddPerturbationEulerScheme(perModelX, perModelY, perModelZ, perTimeArray, N, 0, 0.0001, "z")
'''

#
# Finding Tau
#

# Finding General Difference
diffX = abs(baseModelX - perModelX)
diffY = abs(baseModelY - perModelY)
diffZ = abs(baseModelZ - perModelZ)

error = 1

# Finding the time tau for a single variable where tau is the time the pertrubed model has diverged from the original model by a set error
tauX = LorenzModel.findTauSingleVariable(baseModelX, perModelX, baseTimeArray, N, error = error)
tauY = LorenzModel.findTauSingleVariable(baseModelY, perModelY, baseTimeArray, N, error = error)
tauZ = LorenzModel.findTauSingleVariable(baseModelZ, perModelZ, baseTimeArray, N, error = error)

# Finding the time tau for the whole system where tau is the time the pertrubed model has diverged from the original model by a set error
error = 0.1
tau, errorValues = LorenzModel.findTau(baseModelX, baseModelY, baseModelZ, perModelX, perModelY, perModelZ, baseTimeArray, N, error = error)

'''
print(f"Tau X: {tauX}")
print(f"Tau Y: {tauY}")
print(f"Tau Z: {tauZ}")
'''

#
# Graphs
#

#Graphing Individual Component Values Of Altered System

'''
Range = 1000
plt.figure()
plt.plot(perTimeArray[:Range], perModelX[:Range], color = "r", label = "x")
plt.plot(perTimeArray[:Range], perModelY[:Range], color = "g", label = "y")
plt.plot(perTimeArray[:Range], perModelZ[:Range], color = "b", label = "z")
plt.legend()
plt.show()
'''

#Graphing Continous State Graph In 3d Space Of Altered System

'''
plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot3D(perModelX, perModelY, perModelZ)
plt.title("State Graph Of Perturbed Lorentz Model")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-axis")
plt.show()
'''

#Plotting General Difference

Range = 4000
plt.figure()
plt.plot(perTimeArray[:Range], diffX[:Range], color = (1, 0, 0, 0.5), label = "x")
plt.plot(perTimeArray[:Range], diffY[:Range], color = (0, 0, 0.8, 0.5), label = "y")
plt.plot(perTimeArray[:Range], diffZ[:Range], color = (0, 0.7, 0, 0.5), label = "z")
plt.legend()
plt.show()


# Error Graph For Given Lead Time

'''
Range = len(errorValues)
plt.figure()
plt.plot(perTimeArray[:Range], errorValues[:Range], color = "r", label = "Error")
plt.legend()
plt.show()
'''
