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

# Finding Data Filea
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
modelType = getattr(MLModels, "RNN_Basic_SIMPLE")

# Lead times to run
# T = [0.1, 0.2, 0.3, 0.4, 0.5] + [n for n in range(1,18)]
# T = [n/10 for n in range(1, 100)]
T = [1, 2, 3]

# Sets individual prediction or entire point prediction
fullPointPrediction = False

# Displays graphs at specified lead times
graphT = [0.2, 0.5, 1.0, 2.0, 3.0]
graphRange = 10000

# Records all MSEs
mseList = []

# Data Size
dataSize = 10000
testSize = 10000

# Sequence Length For RNNs
SEQ_LEN = 10

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

# Saving Models
if (not (modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"))):
    if (modelType.__name__.startswith("FNN")):
        modelsPath = parent_dir + f"\\Data\\ML Models Data\\NN Data\\FNN Data\\Direct Prediction\\{modelType.__name__} Data\\Models"
    else:
        modelsPath = parent_dir + f"\\Data\\ML Models Data\\NN Data\\RNN Data\\Direct Prediction\\{modelType.__name__} Data\\Models"
    if not os.path.exists(modelsPath):
        os.makedirs(modelsPath)
    else:
        clearDirectory(modelsPath)

for tau in T:

    print(f"Running lead time {tau}...")
    
    shift = int(tau/0.01) # 0.01 is the time step of the file loaded

    # Creating inputs and labels for training model
    if modelType.__name__.startswith("RNN"):
        trainInputs, _, trainLabels, _ = MLModels.splitDataSequentially(normArray, shift, SEQ_LEN=SEQ_LEN, test_split=0.05, dataSize=dataSize)
        _, testInputs, _, testLabels = MLModels.splitDataSequentially(normArray[dataSize:], shift, SEQ_LEN=SEQ_LEN, test_split=0.95, dataSize=testSize)
    else:
        trainInputs, _, trainLabels, _ = MLModels.splitData(normArray, shift, 0.05, dataSize=dataSize)
        _, testInputs, _, testLabels = MLModels.splitData(normArray[dataSize:], shift, 0.95, dataSize=testSize)

    if not fullPointPrediction:

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
            model = modelType(trainInputs, fullPointPred=fullPointPrediction)

            # Trains Model
            if modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"):
                model.fit(trainInputs, labelPair[0])
            else:
                model.fit(trainInputs, labelPair[0], epochs=100, batch_size=512, verbose = 1) # Verbose is for progress checking. Set value to 1 to turn on

            # Prediction
            if modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"):
                predicted = model.predict(testInputs)
            else:
                predicted = model.predict(testInputs, verbose = 0) # Verbose is for progress checking. Set value to 1 to turn on


            errorMSE = MLModels.metrics.mean_squared_error(predicted, labelPair[1])

            print(errorMSE)

            # Saving Data
            mseTuple.append(errorMSE)

            predictedHistories.append(predicted)

            # Saving models
            if (not (modelType.__name__.startswith("SVR") or modelType.__name__.startswith("Lin"))):
                model.save(f"{modelsPath}//{modelType.__name__}_{tau}_{labelPair[2]}_Model.keras")


        mseList.append(mseTuple)

        predictedHistories = np.array(predictedHistories).reshape(len(predictedHistories[0]), 3)

    else:

        # Creates Model
        model = modelType(trainInputs, fullPointPred=fullPointPrediction)
        
        # Trains Model
        model.fit(trainInputs, trainLabels, epochs=100, batch_size=512, verbose = 1) # Verbose is for progress checking. Set value to 1 to turn on

        # Prediction
        predicted = model.predict(testInputs, verbose = 0)

        errorMSE = MLModels.metrics.mean_squared_error(predicted, testLabels)

        # Saves Predicted Data For Graphing
        predictedHistories = predicted

        mseList.append(errorMSE)
        
        # Saves Model For Further Use
        model.save(f"{modelsPath}//{modelType.__name__}_{tau}_Model.keras")

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