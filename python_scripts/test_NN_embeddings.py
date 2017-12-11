#-----------------------------------------
test_NN.py
#-----------------------------------------

#----------------------------------------
# Set-up
 
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

#-----------------------------------------
# Load Data
# ( adapted from test_algorithms.py)

f = '../features/1st_batch_prepro_min0_size250_win1_2018-EI-reg-es-fear-dev.csv'
dataset = np.loadtxt(f, delimiter=",", skiprows = 1)

X = dataset[:,0:-1] #select everything but last column (label)
Y = dataset[:,-1]   #select column

#-----------------------------------------
# Keras model set-up
# ( Current settings arbitary )

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X.shape[1],))) 
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy']) # accuracy required

#-----------------------------------------
# Model fit

model.fit(X, Y, epochs=9, batch_size=32, verbose=2) # verbose indicates level of detail in process printing
