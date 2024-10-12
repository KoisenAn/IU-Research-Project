#
#Importing libraries
#

#python -m pip install -U matplotlib
import matplotlib.pyplot as plt
#python -m pip install -U numpy
import numpy as np
#python -m pip install pandas
import pandas as pd
import math
import os

#python -m pip install tensorflow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, LSTM, BatchNormalization

#pip install -U scikit-learn
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error


#
#
# Functions
#
#

#Organizing Training/Testing Data For Linear Regression, SVM, FNNs
def splitData(data: list, shift: int, test_split: float, dataSize: int):

    df = pd.DataFrame(data = data[:dataSize], index = range(dataSize), columns = list("XYZ"))
    df.fillna(-99999,inplace = True)

    df["LabelX"] = df["X"].shift(-shift)
    df["LabelY"] = df["Y"].shift(-shift)
    df["LabelZ"] = df["Z"].shift(-shift)
    df.dropna(inplace = True)

    inputs = df.drop(["LabelX","LabelY","LabelZ"],axis = 1)
    labels = df.drop(["X","Y","Z"], axis = 1)

    # Splits the data into training/validation.
    # The data is shuffled by default (shuffle = true).
    traindata, testdata, trainlabel, testlabel = model_selection.train_test_split(inputs, labels, test_size = 0.5, shuffle=False)
    return traindata, testdata, trainlabel, testlabel

#Organizing Training/Testing Data For RNNs
def splitDataSequentially(data: list, leadTime: int, SEQ_LEN: int, test_split: float, dataSize: int) -> list:

    df = pd.DataFrame(data = data[:dataSize], index = range(dataSize), columns = list("XYZ"))
    df.fillna(-99999, inplace = True)

    # Creates the labels
    df["LabelX"] = df["X"].shift(-leadTime)
    df["LabelY"] = df["Y"].shift(-leadTime)
    df["LabelZ"] = df["Z"].shift(-leadTime)
    df.dropna(inplace = True)

    # Creates the sequences for input
    dfSequences = pd.DataFrame()
    for i in range(SEQ_LEN):
        dfSequences[f"Tuple Shift X {i}"] = df["X"].shift(i)
        dfSequences[f"Tuple Shift Y {i}"] = df["Y"].shift(i)
        dfSequences[f"Tuple Shift Z {i}"] = df["Z"].shift(i)
        df.dropna(inplace = True)

    X = dfSequences.values[SEQ_LEN:].reshape(len(dfSequences.values[SEQ_LEN:]), SEQ_LEN, 3)
    y = df[["LabelX", "LabelY", "LabelZ"]][SEQ_LEN:]

    # Splits the data into training/validation.
    # The data is shuffled by default (shuffle = true).
    traindata, testdata, trainlabel, testlabel = model_selection.train_test_split(X, y, test_size = test_split, shuffle = False)
    return traindata, testdata, trainlabel, testlabel

# TODO: Implement a logistic regression algorithm to predict the saturation time.
#def findSaturationTime():

#
#
# Models
#
#

#
# SVR
#

def SVR():
    model = svm.SVR(max_iter=1000)
    return model

def SVR_RBF():
    model = svm.SVR(kernel='rbf')
    return model

def SVR_Linear():
    model = svm.SVR(kernel='linear')
    return model

def SVR_Poly():
    model = svm.SVR(kernel='poly')
    return model

def SVR_Sigmoid():
    model = svm.SVR(kernel='sigmoid')
    return model

def SVR_Precomputed():
    model = svm.SVR(kernel='precomputed')
    return model

#
# Linear Regression
#

def LinReg():
    model = LinearRegression()
    return model

def LinReg_L1():
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    return model

#
# Feed Forward Neural Networks
#

def FNN_Basic(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Example of use: model = FNN_Custom(data, DNNList = [(64, "relu", 0.1), (64, "relu", 0.1), (32, "relu", 0.1)], fullPointPred = True)
def FNN_Custom(input, FNNLayers: list, fullPointPred = False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
     
    for infoTuple in FNNLayers:
        model.add(Dense(infoTuple[0], activation = infoTuple[1]))
        model.add(Dropout(infoTuple[2]))
        model.add(BatchNormalization()) 

    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Different Layer Count

def FNN_3Layers(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_4Layers(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_5Layers(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_6Layers(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Different Layer Density

def FNN_64Dense(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_128Dense(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_256Dense(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 256, activation = "relu"))
    model.add(Dense(units = 256, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_512Dense(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 512, activation = "relu"))
    model.add(Dense(units = 512, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def FNN_DenseIncrease(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_DenseIncreaseDecrease(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_DenseDecrease(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 16, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Different Optimizer functions

def FNN_AdamOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def FNN_SGDOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

def FNN_AdaDeltaOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adadelta', loss='mse', metrics=['mae'])
    return model

# Different Activation functions

def FNN_SoftmaxAct(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "softmax"))
    model.add(Dense(units = 32, activation = "softmax"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def FNN_SigmoidAct(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "sigmoid"))
    model.add(Dense(units = 32, activation = "sigmoid"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_TanhAct(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "tanh"))
    model.add(Dense(units = 32, activation = "tanh"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Custom FFNN Models

def FNN_7Layers_256Dense_Dropout(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def FNN_5Layer_AdamOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 32, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def FNN_IncreasingLayers_AdamOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def FNN_IncreasingLayers256_AdamOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dense(units = 256, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def FNN_IncreasingLayers512_AdamOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dense(units = 256, activation = "relu"))
    model.add(Dense(units = 512, activation = "relu"))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def FNN_IncreasingLayers512_AdamOp_Dropout(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(units = 64, activation = "relu"))
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 256, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 512, activation = "relu"))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#
# Recurrent Neural Networks
# 

def RNN_Basic_SIMPLE(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(SimpleRNN(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_Basic_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# Example of use: model = RNN_Custom(data, RNNLayers = [(32, 0.1), (64, 0.2), (128, 0.2)], DNNList = [(64, "relu", 0.1), (64, "relu", 0.1), (32, "relu", 0.1)], fullPointPred = True)
def RNN_Custom(input, RNNLayers: list, FNNLayers: list, fullPointPred = False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    
    if (len(RNNLayers) > 1):
        for infoTuple in RNNLayers[:-1]:
            model.add(LSTM(infoTuple[0], return_sequences=True))
            model.add(Dropout(infoTuple[1]))
            model.add(BatchNormalization())

    model.add(LSTM(RNNLayers[-1][0], return_sequences=False))
    model.add(Dropout(RNNLayers[-1][1]))
    model.add(BatchNormalization())
     
    for infoTuple in FNNLayers:
        model.add(Dense(infoTuple[0], activation = infoTuple[1]))
        model.add(Dropout(infoTuple[2]))
        model.add(BatchNormalization()) 

    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Different Layer Count

def RNN_2Layer_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_3Layer_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_4Layer_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_5Layer_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_1Layer_LSTM_1Layer_FNN(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_1Layer_LSTM_2Layer_FNN(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_1Layer_LSTM_3Layer_FNN(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_2Layer_LSTM_1Layer_FNN(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())     
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_2Layer_LSTM_2Layer_FNN(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())     
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_2Layer_LSTM_3Layer_FNN(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())     
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Layer Density

def RNN_64Dense_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_128Dense_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_256Dense_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def RNN_512Dense_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Different Optimizer Functions

def RNN_AdamOp_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def RNN_SGDOp_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

def RNN_AdaDeltaOp_LSTM(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adadelta', loss='mse', metrics=['mae'])
    return model

# Custom RNN Models

def RNN_2Layer_128Dense_LSTM_FNN_IncreasingLayers_AdamOp(input, fullPointPred=False):
    model = keras.Sequential()
    model.add(keras.Input(shape=(np.shape(input)[1],)))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) 
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(units = 128, activation = "relu"))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(units = 256, activation = "relu"))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    if (fullPointPred):
        model.add(Dense(units = 3))
    else:
        model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model