#python -m pip install -U matplotlib
import matplotlib.pyplot as plt
#python -m pip install -U numpy
import numpy as np
#python -m pip install pandas
import pandas as pd
import math

def forcingX(x,y,sigma):
    f = sigma * (y - x)
    return f

def forcingY(x,y,z,rho):
    f = x * (rho - z) - y
    return f

def forcingZ(x,y,z,beta):
    f = x * y - beta * z
    return f

# Generates the Lorenz system with the classical Euler scheme
def GenerateEulerScheme(x, y, z, timeArray, N, dt = 0.01, sigma = 10, rho = 28, beta = 8/3):

    for i in range(N-1):
        x[i + 1] = x[i] + forcingX(x[i],y[i],sigma) * dt
        y[i + 1] = y[i] + forcingY(x[i],y[i],z[i],rho) * dt
        z[i + 1] = z[i] + forcingZ(x[i],y[i],z[i],beta) * dt
        timeArray[i + 1] = timeArray[i] + dt

    return x, y, z, timeArray

# Generates the Lorenz system with the classical Euler scheme but adds a perturbation at a given index
def AddPerturbationEulerScheme(x, y, z, timeArray, N, index, perturbation, variable = "x", dt = 0.01, sigma = 10, rho = 28, beta = 8/3):
    for i in range(N-1):
        if i == index:
            if (variable == "x"):
                x[index] = x[index] + perturbation
            elif (variable == "y"):
                y[index] = y[index] + perturbation
            elif (variable == "z"):
                z[index] = z[index] + perturbation
            else:
                print("Error: No dimension found")
                return

        x[i + 1] = x[i] + forcingX(x[i],y[i],sigma) * dt
        y[i + 1] = y[i] + forcingY(x[i],y[i],z[i],rho) * dt
        z[i + 1] = z[i] + forcingZ(x[i],y[i],z[i],beta) * dt
        timeArray[i + 1] = timeArray[i] + dt

    return x, y, z, timeArray

def GenerateCentralScheme(x, y, z, timeArray, N, dt = 0.01, sigma = 10, rho = 28, beta = 8/3):
    for i in range(N-1):
        x[i + 2] = x[i] + forcingX(x[i+1],y[i+1],sigma) * 2 * dt
        y[i + 2] = y[i] + forcingY(x[i+1],y[i+1],z[i+1],rho) * 2 * dt
        z[i + 2] = z[i] + forcingZ(x[i+1],y[i+1],z[i+1],beta) * 2 * dt
        timeArray[i + 1] = timeArray[i] + dt
    return x, y, z, timeArray

def normalize(x, y = None, z = None):
    if (y == None and z == None):
        return x/np.std(x)
    elif (z == None):
        return x/np.std(x), y/np.std(y)
    else:
        return x/np.std(x), y/np.std(y), z/np.std(z)

# This function is for finding saturation time
# This function only works when there are many data points and the time step is small
def findSaturationTime(x, timeArray, error = 0.01, range = 1000, overError = 1):
    i = 0
    pastValues = []
    while True:
        i += 1
        pastValues.append(x[i])
        if (len(pastValues) == range):
            
            del pastValues[0]

            average = sum(pastValues)/len(pastValues)

            if (average > overError and abs(x[i+1]-average) < error):
                return timeArray[i]
            
        if (i == len(x)-2):
            return "Error! Saturation Time Not Found"

# This function is for determining the time value tau when two time series diverge for a single variable
def findTauSingleVariable(x1, x2, timeArray, N, error = 1):
    i = 0
    while abs(x2[i]-x1[i]) < error:
        i += 1
        if i == N:
            print("Error! Tau Not Found")
            return False
    return timeArray[i]

# This function is for determining the time value tau when two systems diverge at a given error
def findTau(baseX, baseY, baseZ, perX, perY, perZ, timeArray, N, error = 1):
    i = 0
    differenceData = []

    difference = math.sqrt((baseX[i]-perX[i])**2 + (baseY[i]-perY[i])**2 + (baseZ[i]-perZ[i])**2)
    while difference < error:
        i += 1
        if i == N:
            print("Error! Tau Not Found")
            return False, differenceData
        
        differenceData.append(difference)
        difference = math.sqrt((baseX[i]-perX[i])**2 + (baseY[i]-perY[i])**2 + (baseZ[i]-perZ[i])**2)

    return timeArray[i], np.array(differenceData)

# Time steps/Increment values
dt = 0.01
T = 10000
N = round(T/dt) + 1

# Data Set
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
timeArray = np.zeros(N)

# Initial Values
x[0] = 0.1
y[0] = 0.1
z[0] = 0.1

# Generate x, y, and z values

GenerateEulerScheme(x, y, z, timeArray, N, dt)

# Generate Normalized Data

xNorm = x/np.std(x)
yNorm = y/np.std(y)
zNorm = z/np.std(z)

'''
# Time series for rate of convection
plt.figure()
plt.plot(timeArray, x, color = "r")
plt.title("Graph of Convection Rate Over Time")
plt.xlabel("Time")
plt.ylabel("Convection Rate")

plt.show()
'''

'''
# Time series for rate of convection
plt.figure()
plt.plot(timeArray, y, color = "b")
plt.title("Graph of Horizontonal Temperature Difference Over Time")
plt.xlabel("Time")
plt.ylabel("Convection Rate")

plt.show()
'''

'''
# Time series for rate of convection
plt.figure()
plt.plot(timeArray, z, color = "g")
plt.title("Graph of Convection Rate Over Time")
plt.xlabel("Time")
plt.ylabel("Convection Rate")

plt.show()
'''

'''
# 3 dimensional state graph of x, y, and z values
plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot3D(x,y,z)
plt.title("State Graph Of Lorentz Model")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-axis")

plt.show()
'''

'''
#Creates Data For The Lorenz System
f = open("lorenzData.txt", "w")

for i in range(len(timeArray)):
    f.write(f"{x[i]} {y[i]} {z[i]}")
    f.write("\n")

f.close()
'''

'''
#Creates Normalized Data For The Lorenz System
f = open("lorenzNormData.txt", "w")

for i in range(len(timeArray)):
    f.write(f"{xNorm[i]} {yNorm[i]} {zNorm[i]}")
    f.write("\n")

f.close()
'''