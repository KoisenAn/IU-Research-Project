#python -m pip install -U matplotlib
import matplotlib.pyplot as plt
#python -m pip install -U numpy
import numpy as np

def forcingX(x,y,sigma):
    f = sigma * (y - x)
    return f

def forcingY(x,y,z,rho):
    f = x * (rho - z) - y
    return f

def forcingZ(x,y,z,beta):
    f = x * y - beta * z
    return f

#Generates the Lorenz system with the classical Euler scheme
def GenerateEulerScheme(x, y, z, timeArray, N, dt = 0.01, sigma = 10, rho = 28, beta = 8/3):

    for i in range(N-1):
        x[i + 1] = x[i] + forcingX(x[i],y[i],sigma) * dt
        y[i + 1] = y[i] + forcingY(x[i],y[i],z[i],rho) * dt
        z[i + 1] = z[i] + forcingZ(x[i],y[i],z[i],beta) * dt
        timeArray[i + 1] = timeArray[i] + dt
    return x, y, z, timeArray

#Generates the Lorenz system with the classical Euler scheme but adds a perturbation at a given index
def AddPerturbationEulerScheme(x, y, z, timeArray, N, index, perturbation, dt = 0.01, sigma = 10, rho = 28, beta = 8/3):
    for i in range(N-1):
        x[i + 1] = x[i] + forcingX(x[i],y[i],sigma) * dt
        y[i + 1] = y[i] + forcingY(x[i],y[i],z[i],rho) * dt
        z[i + 1] = z[i] + forcingZ(x[i],y[i],z[i],beta) * dt
        timeArray[i + 1] = timeArray[i] + dt

        if i == index - 1:
            x[index] = x[index] + perturbation

    return x, y, z, timeArray

def GenerateCentralScheme(x, y, z, timeArray, N, dt = 0.01, sigma = 10, rho = 28, beta = 8/3):
    for i in range(N-1):
        x[i + 2] = x[i] + forcingX(x[i+1],y[i+1],sigma) * 2 * dt
        y[i + 2] = y[i] + forcingY(x[i+1],y[i+1],z[i+1],rho) * 2 * dt
        z[i + 2] = z[i] + forcingZ(x[i+1],y[i+1],z[i+1],beta) * 2 * dt
        timeArray[i + 1] = timeArray[i] + dt
    return x, y, z, timeArray

# TODO: Fix function below. We do not define tau to be the time when two series diverge but when one saturates with error.
# This function is for determining the time value tau when two time series diverge
def findTau(x1, x2, timeArray, N, error):
    i = 0
    while abs(x2[i]-x1[i]) < error:
        i += 1
    return timeArray[i]

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
plt.plot(timeArray1, x, color = "r")
plt.title("Graph of Convection Rate Over Time")
plt.xlabel("Time")
plt.ylabel("Convection Rate")

plt.show()
'''

'''
# Time series for rate of convection
plt.figure()
plt.plot(timeArray1, y, color = "b")
plt.title("Graph of Horizontonal Temperature Difference Over Time")
plt.xlabel("Time")
plt.ylabel("Convection Rate")

plt.show()
'''

'''
# Time series for rate of convection
plt.figure()
plt.plot(timeArray1, z, color = "g")
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


#Creates Normalized Data For The Lorenz System
f = open("lorenzNormData.txt", "w")

for i in range(len(timeArray)):
    f.write(f"{xNorm[i]} {yNorm[i]} {zNorm[i]}")
    f.write("\n")

f.close()
