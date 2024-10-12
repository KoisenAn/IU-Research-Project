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

# Finding Data File

import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the file in the parent directory
parent_dir = os.path.dirname(current_dir)

import sys
 
# Appending the directory of lorenzNormData.txt, and MLModels.py in the sys.path list
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
modelType = getattr(MLModels, "LinReg")

# Lead times to run
# T = [0.1, 0.2, 0.3, 0.4, 0.5] + [n for n in range(1,18)]
T = [n/10 for n in range(1, 100)]
# T = [1, 2, 3]

# Sets individual prediction or entire point prediction
fullPointPrediction = False

# Displays graphs at specified lead times
graphT = [0.2, 0.5, 1.0, 2.0, 3.0]
graphRange = 10000

# Records all MSEs
mseList = []

# Data Size
dataSize = 10000

# Creating folder to store files
# TODO: Make seperate files for full point and single variable prediction

if modelType.__name__.startswith("SVR"):
    if fullPointPrediction:
        print("Error: SVR models cannot predict entire points")
        sys.exit()
    path = parent_dir + f"\\Data\\ML Models Data\\SVR Data\\Direct Prediction\\{modelType.__name__} Data"
elif modelType.__name__.startswith("Lin"):
    if fullPointPrediction:
        print("Error: Linear Regression models cannot predict entire points")
        sys.exit()
    path = parent_dir + f"\\Data\\ML Models Data\\Linear Regression Data\\Direct Prediction\\{modelType.__name__} Data"
else:
    if modelType.__name__.startswith("FNN"):
        path = parent_dir + f"\\Data\\ML Models Data\\NN Data\\FNN Data\\Direct Prediction\\{modelType.__name__} Data"
    elif modelType.__name__.startswith("RNN"):
        path = parent_dir + f"\\Data\\ML Models Data\\NN Data\\RNN Data\\Direct Prediction\\{modelType.__name__} Data"
    else:
        print("Error: Model Not Identified")
        sys.exit()

if not os.path.exists(path):
    os.makedirs(path)
else:
    clearDirectory(path)


for tau in T:

    print(f"Running lead time {tau}...")

    if not fullPointPrediction:

        shift = int(tau/0.01) # 0.01 is the time step of the file loaded

        # Creating inputs and labels for training model
        trainInputs, testInputs, trainLabels, testLabels = MLModels.splitData(normArray, shift, 0.05, dataSize=dataSize)

        # Creating labels for each component during both training and verification
        trainLabelX = trainLabels.drop(["LabelY","LabelZ"],axis = 1)
        trainLabelY = trainLabels.drop(["LabelX","LabelZ"],axis = 1)
        trainLabelZ = trainLabels.drop(["LabelX","LabelY"],axis = 1)

        testLabelX = testLabels.drop(["LabelY","LabelZ"],axis = 1)
        testLabelY = testLabels.drop(["LabelX","LabelZ"],axis = 1)
        testLabelZ = testLabels.drop(["LabelX","LabelY"],axis = 1)

        # Saves X, Y and Z MSEs
        mseTuple = []

        # Saves Predicted Data For Graphing
        predictedHistories = []

        for labelPair in [(trainLabelX, testLabelX, "X"), (trainLabelY, testLabelY, "Y"), (trainLabelZ, testLabelZ, "Z")]:

            # Creates Model
            model = modelType(trainInputs.to_numpy(), fullPointPred=fullPointPrediction)

            # Trains Model

            if modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"):
                model.fit(trainInputs.to_numpy(), labelPair[0].to_numpy())
            else:
                model.fit(trainInputs.to_numpy(), labelPair[0].to_numpy(), epochs=100, batch_size=512, verbose = 1) # Verbose is for progress checking. Set value to 1 to turn on

            # Prediction
            if modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"):
                predicted = model.predict(testInputs.to_numpy())
            else:
                predicted = model.predict(testInputs.to_numpy(), verbose = 0) # Verbose is for progress checking. Set value to 1 to turn on


            errorMSE = MLModels.metrics.mean_squared_error(predicted, labelPair[1])

            print(errorMSE)

            # Saving Data
            mseTuple.append(errorMSE)

            predictedHistories.append(predicted)

            # Saving models
            if (not (modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"))):
                path = parent_dir + f"\\Data\\ML Models Data\\SVR Data\\Direct Prediction\\{modelType.__name__} Data\\Models"
                clearDirectory(path)

                model.save(f"{path}//{modelType.__name__}_{tau}_{labelPair[2]}_Model.keras")


        mseList.append(mseTuple)

        predictedHistories = np.array(predictedHistories).reshape(len(predictedHistories[0]), 3)

    else:
        
        shift = int(tau/0.01) # 0.01 is the time step of the file loaded

        # Creating inputs and labels for training model
        trainInputs, testInputs, trainLabels, testLabels = MLModels.splitData(normArray, shift, 0.05, dataSize=dataSize)

        model = modelType(trainInputs.to_numpy(), fullPointPred=fullPointPrediction)
        
        model.fit(trainInputs.to_numpy(), trainLabels.to_numpy(), epochs=100, batch_size=512, verbose = 1) # Verbose is for progress checking. Set value to 1 to turn on

        # Prediction
        predicted = model.predict(testInputs.to_numpy(), verbose = 0)

        errorMSE = MLModels.metrics.mean_squared_error(predicted, testLabels)

        # Saves Predicted Data For Graphing
        predictedHistories = predicted

        mseList.append(errorMSE)
        
        # Saves Model For Further Use
        model.save(f"{path}//{modelType.__name__}_{tau}_Model.keras")

    #print(errorMSE)

    ''''
    # Graphing
    # NOTE: Program will stop for each graph
    if (tau in graphT):

        plt.figure(figsize=(12,12))
        ax = plt.axes(projection='3d')

        ax.grid()
        ax.plot3D(predictedHistories[:graphRange,0], predictedHistories[:graphRange,1], predictedHistories[:graphRange,2])
        
        plt.title("State Graph Of Lorenz Model")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_zlabel("Z-axis")

        plt.show()
    '''

    print(f"Completed lead time {tau}")

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