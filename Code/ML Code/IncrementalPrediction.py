#
# Importing Libraries
#

#python -m pip install -U matplotlib
import matplotlib.pyplot as plt
#python -m pip install -U numpy
import numpy as np
#python -m pip install pandas
import pandas as pd
import math
import os

#
# Finding data files and other classes
#

import os

# Get the current script's Directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the file in the parent Directory
parent_dir = os.path.dirname(current_dir)

import sys
 
# Appending the Directory of lorenzNormData.txt, and MLModels.py in the sys.path list
sys.path.append(parent_dir+"\\Data\\Lorenz Model Data") 
sys.path.append(parent_dir+"\\ML Code")
sys.path.append(parent_dir+"\\Lorenz Model Code")

def clearDirectory(path):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting/overriding files.")

# Loading Data File and accessing ML Model Class
import MLModels
file = open(parent_dir+"\\Data\\Lorenz Model Data\\lorenzNormData.txt", "r")
normArray = np.loadtxt(file)

'''
# Checking Data

dataRange = 10000

plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot3D(normArray[:dataRange,0], normArray[:dataRange,1], normArray[:dataRange,2])
plt.title("State Graph Of Lorenz Model")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-axis")

plt.show()
'''

#
# Initial parameters
#

# Setting Model Type
modelType = getattr(MLModels, "FNN_Basic")

# Lead times to run
# T = [0.1, 0.2, 0.3, 0.4, 0.5] + [n for n in range(1,18)]
# T = [n/10 for n in range(1, 100)]
T = [1.0, 2.0, 3.0]

# Sets individual prediction or entire point prediction
fullPointPrediction = False

# Displays graphs at specified lead times
graphT = [1.0]
# graphT = [0.2, 0.5, 1.0, 2.0, 3.0]
graphRange = 10000

# Records all MSEs
mseList = []

# Data Size
dataSize = 10000
testSize = 10000

# Sequence Length For RNNs
SEQ_LEN = 10

# Time Step
timeStep = 1

# Creating folder to store files
# TODO: Make seperate files for full point and single variable prediction

if modelType.__name__.startswith("SVR"):
    if fullPointPrediction:
        print("Error: SVR models cannot predict entire points")
        sys.exit()
    path = parent_dir + f"\\Data\\ML Models Data\\SVR Data\\Incremental Prediction\\{modelType.__name__} Data"
elif modelType.__name__.startswith("Lin"):
    if fullPointPrediction:
        print("Error: Linear Regression models cannot predict entire points")
        sys.exit()
    path = parent_dir + f"\\Data\\ML Models Data\\Linear Regression Data\\Incremental Prediction\\{modelType.__name__} Data"
elif modelType.__name__.startswith("FNN"):
    path = parent_dir + f"\\Data\\ML Models Data\\NN Data\\FNN Data\\Incremental Prediction\\{modelType.__name__} Data"
else:
    print("Error: Model Not Identified")
    sys.exit()

if not os.path.exists(path):
    os.makedirs(path)
else:
    clearDirectory(path)

#
# Organizing Data For ML Models/Creates And Trains Models
#

trainInputs, _, trainLabels, _ = MLModels.splitData(normArray, timeStep, 0.05, dataSize=dataSize)
_, testInputs, _, testLabels = MLModels.splitDataIncrementalPrediction(normArray[dataSize:], T, 0.95, dataSize=testSize)

if not fullPointPrediction:

    # Creating labels for each component during both training and verification
    trainLabelX = trainLabels.drop(["LabelY","LabelZ"], axis = 1)
    trainLabelY = trainLabels.drop(["LabelX","LabelZ"], axis = 1)
    trainLabelZ = trainLabels.drop(["LabelX","LabelY"], axis = 1)
    
    # Creates Model
    modelX = modelType(trainInputs, fullPointPred=fullPointPrediction)
    modelY = modelType(trainInputs, fullPointPred=fullPointPrediction)
    modelZ = modelType(trainInputs, fullPointPred=fullPointPrediction)

    # Trains Model
    if modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"):
        modelX.fit(trainInputs, trainLabelX)
        modelY.fit(trainInputs, trainLabelY)
        modelZ.fit(trainInputs, trainLabelZ)
    else:
        modelX.fit(trainInputs, trainLabelX, epochs=100, batch_size=512, verbose = 1) # Verbose is for progress checking. Set value to 1 to turn on
        modelY.fit(trainInputs, trainLabelY, epochs=100, batch_size=512, verbose = 1)
        modelZ.fit(trainInputs, trainLabelZ, epochs=100, batch_size=512, verbose = 1)

else:

    # Creates Model
    model = modelType(trainInputs, fullPointPred=fullPointPrediction)
    
    # Trains Model
    model.fit(trainInputs, trainLabels, epochs=100, batch_size=512, verbose = 1) # Verbose is for progress checking. Set value to 1 to turn on

#
# Incremental Prediction 
#

currInputs = testInputs
for i in range(int(max(T)/0.01)+1):
    
    tau = i * 0.01
    print(f"Predicting lead time {tau}...")

    if not fullPointPrediction:

        predictX = modelX.predict(currInputs)
        predictY = modelY.predict(currInputs)
        predictZ = modelZ.predict(currInputs)

        newInputs = pd.DataFrame(data = np.column_stack((predictX, predictY, predictZ)),
                                     index = range(len(predictX)), columns = list("XYZ"))
        newInputs.fillna(-99999,inplace = True)
        currInputs = newInputs

    else:
            
        predicted = model.predict(currInputs)

        currInputs = predicted

    if tau in T:

        testLabelsX = testLabels[f"LabelX {tau}"]
        testLabelsY = testLabels[f"LabelY {tau}"]
        testLabelsZ = testLabels[f"LabelZ {tau}"]

        if not fullPointPrediction:
            errorMSEX = MLModels.metrics.mean_squared_error(predictX, testLabelsX)
            errorMSEY = MLModels.metrics.mean_squared_error(predictY, testLabelsY)
            errorMSEZ = MLModels.metrics.mean_squared_error(predictZ, testLabelsZ)

            mseList.append([errorMSEX, errorMSEY, errorMSEZ])

            print(mseList)

        else:
            testLabelsAtTimeTau = pd.DataFrame(data = np.column_stack((testLabelsX, testLabelsY, testLabelsZ)),
                                               index = range(len(testLabelsX)), columns = list("XYZ"))

            errorMSE = MLModels.metrics.mean_squared_error(predicted, testLabelsAtTimeTau)

            mseList.append(errorMSE)

            print(errorMSE)

    # Graphing
    # NOTE: Program will stop for each graph

    if (tau in graphT):

        plt.figure(figsize=(12,12))
        ax = plt.axes(projection='3d')

        ax.grid()
        ax.plot3D(currInputs["X"], currInputs["Y"], currInputs["Z"])
        
        plt.title("State Graph Of Lorenz Model")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_zlabel("Z-axis")

        plt.show()

# Saving Data To File
f = open(f"{path}\\{modelType.__name__} MSE Data.txt", "w")
if fullPointPrediction:
    f.write(f"Full Point Prediction: Yes")
    f.write("\n")
else:
    f.write(f"Full Point Prediction: No")
    f.write("\n")
f.write(f"Lead Times: {T}")
f.write("\n")
f.write(f"MSE List: {mseList}")
f.close()