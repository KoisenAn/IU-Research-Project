#python -m pip install -U matplotlib
import matplotlib.pyplot as plt
#python -m pip install -U numpy
import numpy as np

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

# Normal Time Step
dt = 0.01
T = 100
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

'''
# Initial Constants
LorenzModel.GenerateEulerScheme(x, y, z, timeArray, N, dt = dt)
'''

'''
# Modified Sigma Constant
LorenzModel.GenerateEulerScheme(x, y, z, timeArray, N, dt = dt, sigma = 9)
'''

'''
# Modified Sigma Constant
LorenzModel.GenerateEulerScheme(x, y, z, timeArray, N, dt = dt, sigma = 11)
'''

'''
# Modified Rho Constant
LorenzModel.GenerateEulerScheme(x, y, z, timeArray, N, dt = dt, rho = 27)
'''

'''
# Modified Rho Constant
LorenzModel.GenerateEulerScheme(x, y, z, timeArray, N, dt = dt, rho = 29)
'''

'''
# Modified Beta Constant
LorenzModel.GenerateEulerScheme(x, y, z, timeArray, N, dt = dt, beta = 5/3)
'''


# Modified Beta Constant
LorenzModel.GenerateEulerScheme(x, y, z, timeArray, N, dt = dt, beta = 11/3)


plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot3D(x,y,z)
plt.title("State Graph Of Altered Constant Lorentz Model")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-axis")

plt.show()