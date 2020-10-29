# -*- coding: utf-8 -*-
"""Copy of simulated study

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12p6xxXYa47oGZ0wwcM7FdGsjD-a8FYZE
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import RNN
#from keras.layers import SimpleRNN
#from keras.layers import GRU
#import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
!pip install croston
from croston import croston
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

#create intermittent data
def simID(obs=60, idi=2, cv2=0.5, level=None):
  if (not level):
    m = np.random.uniform(low=9, high=99, size=1)
  else:
    m = level - 1
  if (cv2 != 0):
    p = (m/(cv2*((m+1)**2)))
    r = m*p/(1-p)
    x = np.random.binomial(n=1, p=1/idi, size=obs) * np.random.negative_binomial(n=r, p=p, size=obs)
  else:
    x = np.random.binomial(n=1, p=1/idi, size=obs) * round(m+1)
  x = x.reshape(-1,1)
  return x   

#lag transform on training
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)   
 
#lag transform on test data
def create_testset(dataset, look_back=1, horizon=28):
  dataX = []
  n = len(trainy)
  for i in range(look_back):
    a = pd.Series(np.empty(horizon), name=i)
    a[:] = np.NaN
    a[0:(i+1)] = trainy[(n-1-i):]
    dataX.append(a)
  dataX = pd.DataFrame(dataX).T
  return dataX

#forecast recursive rnn
def forecast(test, model):
  pred = [[0]*28]
  na_condition = test.isna()
  testx_copy = copy.copy(test)
  for i in range(horizon):
    for j in range(look_back):
      if na_condition.iloc[i,j]==True:
        testx_copy.iloc[i,j] = pred[0][i-j]
    value = testx_copy.iloc[:(i+1),:]
    value = value.values
    value = np.reshape(value, ((i+1), 1, look_back))
    result = model.predict(value, batch_size=batch_size)
    pred[0][i] = result[i]
  pred = np.reshape(pred, (horizon, 1))
  return pred

#forecast recursive rnn with n timesteps
def forecast2(test, model):
  pred = [[0]*28]
  na_condition = test.isna()
  testx_copy = copy.copy(test)
  for i in range(horizon):
    for j in range(look_back):
      if na_condition.iloc[i,j]==True:
        testx_copy.iloc[i,j] = pred[0][i-j]
    value = testx_copy.iloc[:(i+1),:]
    value = value.values
    value = np.reshape(value, ((i+1), look_back, 1))
    result = model.predict(value, batch_size=batch_size)
    pred[0][i] = result[i]
  pred = np.reshape(pred, (horizon, 1))
  return pred

#forecast recursive dnn
def forecast_dnn(test, model):
  pred = [[0]*28]
  na_condition = test.isna()
  testx_copy = copy.copy(test)
  for i in range(horizon):
    for j in range(look_back):
      if na_condition.iloc[i,j]==True:
        testx_copy.iloc[i,j] = pred[0][i-j]
    value = testx_copy.iloc[:(i+1),:]
    value = value.values
    value = np.reshape(value, ((i+1), look_back))
    result = model.predict(value, batch_size=batch_size)
    pred[0][i] = result[i]
  pred = np.reshape(pred, (horizon,1))
  return pred
  
#scaler
def scaler(data):
  maxdata = data.max()
  mindata = data.min()
  result = (data-mindata)/(maxdata-mindata)
  return result

#inverse scaler
def inverse_scaler(data1, datainv):
  maxdata = data1.max()
  mindata = data1.min()
  result = (datainv*(maxdata-mindata))+mindata
  return result

#rmsse
def rmsse(train, test, predict, horizon):
  n = len(train)
  nominator = sum((test-predict)**2)
  denominator = sum((train[1:972]-train[:971])**2)
  result = math.sqrt((1/horizon)*(nominator/((1/(n-1))*denominator)))
  return result

#set up the parameter for generating the data
obs = 10000
idi = 10
level = 2
series = simID(obs=obs, idi=idi, level=level)
plt.plot(series)

verbose = 0
epochs = 500
batch_size = 32
batch_size_dnn = 32
horizon = 28
look_back = 28
input_shape_dnn = (look_back,)
input_shape = (1,look_back)
rmssernn = []
rmssecroston = []
rmsseses = []
rmssednn = []
maernn = []
maecroston = []
maeses = []
maednn = []
for i in range(50):
  series = simID(obs=obs, idi=idi, level=level)
  train = series[:-horizon]
  test = series[-horizon:]
  train_scaler = scaler(train)
  trainx, trainy = create_dataset(train_scaler, look_back=look_back)
  trainx_dnn, trainy_dnn = create_dataset(train_scaler, look_back=look_back)
  testx = create_testset(trainy, look_back=look_back)
  trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
  #deep learning model SimpleRNN with timestep 1 and lag as variable
  activation = ('relu', 'sigmoid')
  for a in activation:
    tf.keras.backend.clear_session()
    globals()['model%s' % a] = tf.keras.models.Sequential()
    globals()['model%s' % a].add(tf.keras.layers.SimpleRNN(units=64, activation = 'relu'))
    globals()['model%s' % a].add(tf.keras.layers.Dropout(0.01))
    globals()['model%s' % a].add(tf.keras.layers.Dense(1, activation=a))
    globals()['model%s' % a].compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0075), loss='mean_squared_error')
    history = globals()['model%s' % a].fit(trainx, trainy, epochs=1000, batch_size=batch_size, verbose=0,
                                           callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5, restore_best_weights = True, mode='auto')])
  #deep neural network
  tf.keras.backend.clear_session()
  model_dnn = tf.keras.models.Sequential()
  model_dnn.add(tf.keras.Input(shape=(look_back,)))
  model_dnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
  model_dnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
  model_dnn.add(tf.keras.layers.Flatten())
  model_dnn.add(tf.keras.layers.Dense(units=1, activation='relu'))
  model_dnn.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0075), loss='mse')
  history_dnn = model_dnn.fit(trainx_dnn, trainy_dnn, epochs=500, batch_size=batch_size_dnn, verbose=0,
                             callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode='auto', restore_best_weights=True)])
  #ses model
  model_ses = SimpleExpSmoothing(train).fit(smoothing_level=0.2)
  #croston model
  fit_crost = croston.fit_croston(train, forecast_length=28, croston_variant='sba')
  #forecast
  predict_rnn = [inverse_scaler(train, forecast(testx, modelsigmoid)), inverse_scaler(train, forecast(testx, modelrelu))]
  predict_dnn = forecast_dnn(testx, model_dnn)
  predict_dnn = inverse_scaler(train, predict_dnn)
  predcrost = fit_crost['croston_forecast']
  predict_ses = model_ses.forecast(28).reshape((28,1))
  rmssernn.append(min([rmsse(train, test, predict_rnn[0], horizon), rmsse(train, test, predict_rnn[1], horizon)]))
  rmssednn.append(rmsse(train, test, predict_dnn, horizon))
  rmssecroston.append(rmsse(train, test, predcrost, horizon))
  rmsseses.append(rmsse(train, test, predict_ses, horizon))
  maernn.append(min([mean_absolute_error(test, predict_rnn[0]), mean_absolute_error(test, predict_rnn[1])]))
  maednn.append(mean_absolute_error(test, predict_dnn))
  maecroston.append(mean_absolute_error(test, predcrost))
  maeses.append(mean_absolute_error(test, predict_ses))

print("rmssernn: " + str(sum(rmssernn)/50))
print("rmssednn: " + str(sum(rmssednn)/50))
print("rmssecroston: " + str(sum(rmssecroston)/50))
print("rmsseses: " + str(sum(rmsseses)/50))
print("maernn: " + str(sum(maernn)/50))
print("maednn: " + str(sum(maednn)/50))
print("maecroston: " + str(sum(maecroston)/50))
print("maeses: " + str(sum(maeses)/50))