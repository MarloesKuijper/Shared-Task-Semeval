import argparse
import keras
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from scipy.stats import pearsonr
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import tensorflow as tf



'''Script that implements an LSTM network for regression
   Currently doesn't learn much but at least it does something'''

## USAGE: python lstm.py -f1 TRAIN_SET -f2 DEV_SET [-scale] [-shuffle]
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", required=True, type=str, help="Train file")
    parser.add_argument("-f2", required=True, type=str, help="Dev file")
    parser.add_argument("-folds", default = 10, type=int, help="Number of folds for cross validation")
    parser.add_argument("-task_meta", required=True, type=str, help="Specify the task and emotion or valence (used in results file)")
    parser.add_argument("-scale", action='store_true', help="If added we scale the features between 0 and 1 (sometimes helps)")
    parser.add_argument("-shuffle", action='store_true', help="If added we shuffle the rows before training (different results for cross validation)")
    args = parser.parse_args()
    return args


def scale_dataset(dataset):
    '''Scale features of dataset between 0 and 1
       This apparantly helps sometimes'''
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset


def load_dataset(f, scale, shuffle):
    dataset = np.loadtxt(f, delimiter=",", skiprows = 1)
    if scale:
        dataset = scale_dataset(dataset)
    if shuffle: #shuffle for more randomization
        np.random.shuffle(dataset)
    X = dataset[:,0:-1] 
    Y = dataset[:,-1] 
    return X, Y, dataset  


def cv_dataset(dataset, low, up):
    '''Apparently Keras has no cross validation functionality built in so do it here
       The rows between low and up are for the validation set, rest is for train set'''
    train_rows = np.concatenate((dataset[:low, :], dataset[up:, :] ), axis=0)
    valid_rows = dataset[low:up, :]
    train_X = train_rows[:,0:-1] 
    train_Y = train_rows[:,-1] 
    valid_X = valid_rows[:,0:-1]
    valid_Y = valid_rows[:,-1] 
    #print (dataset.shape, train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape)
    return train_X, train_Y, valid_X, valid_Y

def train_lstm(train_X, train_Y, dev_X, dev_Y, input_dim, nodes):
    # needs -scale
    ## Create model
    model = Sequential()
    model.add(Embedding(input_dim, output_dim=nodes))
    model.add(LSTM(nodes,return_sequences=True))  
    model.add(Dropout(0.25))
    model.add(LSTM(nodes))  # return a single vector of dimension: nodes
    model.add(Dense(1, activation='sigmoid'))                       #last layer should output only a single value since we do regression
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])    #mse as loss and metric, should be good enough even though it's not pearson

    ## Train model
    model.fit(train_X, train_Y, batch_size=16, epochs=1, validation_split = 0.1, verbose=1)
    
    ## Make predictions and evaluate
    pred = model.predict(dev_X, batch_size=16, verbose=0)
    predictions = [p[0] for p in pred] #put in format we can evaluate, avoid numpy error
    print('Score: {0}'.format(round(pearsonr(predictions, dev_Y)[0],4)))

def train_CNN(train_X, train_Y, dev_X, dev_Y, input_dim, nodes):
    # needs -scale
    model = Sequential()
    model.add(Embedding(input_dim, output_dim=nodes))
    model.add(Dropout(0.2))
    model.add(Conv1D(nodes, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(nodes))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(train_X, train_Y, batch_size=16, epochs=10, validation_split=0.1, verbose=0)
    pred = model.predict(dev_X, batch_size=16, verbose=1)
    predictions = [p[0] for p in pred]
    score = round(pearsonr(predictions, dev_Y)[0],4)
    print("score {0}".format(score))
  
    return score

def train_feedforward(train_X, train_Y, dev_X, dev_Y, input_dim, nodes):
    '''Feedforward network similar to Approach 1 of the winners of the WASSA emotion intensity shared task
       http://www.aclweb.org/anthology/W17-5207'''
    ## Create model
    model = Sequential()
    model.add(Dense(nodes[0], input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.50))
    model.add(Dense(nodes[1], activation="relu"))
    model.add(Dense(nodes[2], activation="relu"))
    model.add(Dense(nodes[3], activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    #Train
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])    #mse as loss and metric, should be good enough even though it's not pearson
    model.fit(train_X, train_Y, batch_size=8, epochs=20, validation_split = 0.1, verbose=0)
    
    #Evaluate
    pred = model.predict(dev_X, batch_size=16, verbose=1)
    predictions = [p[0] for p in pred] #put in format we can evaluate, avoid numpy error
    score = round(pearsonr(predictions, dev_Y)[0],4)
    print('Score: {0}'.format(score))
    return score

def train_lstm2(train_X, train_Y, dev_X, dev_Y, input_dim, nodes, batch_size, epochs, optimizer, lstm_layers, lstm_activation, dense_layers, dropout):
    # don't use -scale here
    ## Create model
    train_X = train_X.reshape((len(train_X), 1, len(train_X[0])))
    dev_X = dev_X.reshape((len(dev_X), 1, len(dev_X[0])))
    model = Sequential()
    neurons = len(dev_Y)
    #model.add(Embedding(input_dim=input_dim, output_dim=nodes))
    model.add(LSTM(neurons, input_shape=(len(train_X[0]), len(train_X[0][0])), dropout=dropout, recurrent_dropout=dropout, return_sequences=True, activation=lstm_activation))  
    #model.add(Dropout(0.25))
    if lstm_layers == 2:
        model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=dropout, activation=lstm_activation)) 
    else: 
        model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=True, activation=lstm_activation)) 
        model.add(LSTM(int(nodes/2), dropout=dropout, recurrent_dropout=dropout, activation=lstm_activation))

    if dense_layers > 0:
        if lstm_layers == 2:
            model.add(Dense(int(nodes/2), activation="relu"))
            if dense_layers > 1:
                model.add(Dense(int(nodes/4), activation="relu"))
        else:
            model.add(Dense(int(nodes/4), activation="relu"))
            if dense_layers > 1:
                model.add(Dense(int(nodes/8), activation="relu"))

    model.add(Dense(1, activation='sigmoid'))                       
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse']) 

    ## Train model
    model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_split = 0.1, verbose=1)
    
    ## Make predictions and evaluate
    pred = model.predict(dev_X, batch_size=batch_size, verbose=0)
    predictions = [p[0] for p in pred] #put in format we can evaluate, avoid numpy error
    print('Score: {0}'.format(round(pearsonr(predictions, dev_Y)[0],4)))
    score = round(pearsonr(predictions, dev_Y)[0], 4)
    return predictions, score
    
if __name__ == "__main__":
    args = create_arg_parser()
    
    #Load data  
    train_X, train_Y, dataset = load_dataset(args.f1, args.scale, args.shuffle)
    dev_X, dev_Y, _ = load_dataset(args.f2, args.scale, args.shuffle)
    
    ##LSTM training 
    nodes = [200, 400, 600, 800]
    dropouts = [0.1, 0.2, 0.3]
    #recurrent_dropout = [0.0, 0.1, 0.2, 0.3]
    # batch_size = [8, 16, 32, 48]
    # epochs = [5, 10, 15, 20, 25]
    #optimizer = ["adam", "nadam", "rmsprop", "adamax"]
    lstm_layers = [3]
    dense_layers = [1,2,3]
    lstm_activations = ["relu"]

    # after getting optimal params for relu, check sigmoid/tanh just to be sure

    for node in nodes:
        for lstm_layer_option in lstm_layers:
            for lstm_activation in lstm_activations:
                for dense_layer_option in dense_layers:
                    for dropout in dropouts:                        
                        scores = []
                        avg_scores = []
                        for fold in range(args.folds):
                            predictions = []
                            for i in range(2):
                                low = int(len(dataset) * fold / args.folds)
                                up = int(len(dataset) * (fold +1) / args.folds)
                                train_X, train_Y, valid_X, valid_Y = cv_dataset(dataset, low, up)
                                pred, score = train_lstm2(train_X, train_Y, valid_X, valid_Y, len(dataset[0]), node, 16, 10, "adam", lstm_layer_option, lstm_activation, dense_layer_option, dropout)
                                predictions.append(pred)
                            scores.append(score)
                            # averaged over 2 runs
                            avg_predictions = [(item + predictions[1][ix])/2 for ix, item in enumerate(predictions[0])]
                            avg_score = round(pearsonr(avg_predictions, valid_Y)[0], 4)
                            avg_scores.append(avg_score)                              

                        print ('Average score for {0}-fold cv: {1}'.format(str(args.folds), str(float(sum(scores)) / len(scores))))
                        with open("../LSTM_RESULTS_{0}.txt".format(args.task_meta), "a") as outfile:
                            outfile.write("Average score for {0}-fold cv: {1} (averaged over 2 runs: {11}), with nodes: {2}, batch_size: {3}, epochs: {4}, lstm layers: {5}, lstm activation: {6}, dense layers: {7}, dropout: {8} and optimizer: {9}, with file: {10}".format(str(args.folds), float(sum(scores)) / len(scores), node, 16, 20, lstm_layer_option, lstm_activation, dense_layer_option, dropout, "adam", args.f1, float(sum(avg_scores))/len(avg_scores)))
                            outfile.write("\n")

    
    
    ## Feed forward network training
    # nodes = [300, 125, 50, 25] #should try some different combinations at some point
    # scores = []
    # for fold in range(args.folds): #do cross-validation
    #   low = int(len(dataset) * fold / args.folds)
    #   up  = int(len(dataset) * (fold +1) / args.folds)
    #   train_X, train_Y, valid_X, valid_Y = cv_dataset(dataset, low, up)
    #   score = train_feedforward(train_X, train_Y, valid_X, valid_Y, len(train_X[0]), nodes)
    #   scores.append(score)
    
    # print ('Average score for {0}-fold cv: {1}'.format(args.folds, float(sum(scores)) / len(scores))) 


    ## cnn
    # nodes = [300, 125, 50, 25]
    # scores = []
    # for fold in range(args.folds): #do cross-validation
    #     low = int(len(dataset) * fold / args.folds)
    #     up  = int(len(dataset) * (fold +1) / args.folds)
    #     train_X, train_Y, valid_X, valid_Y = cv_dataset(dataset, low, up)
    #     print(train_X.shape)
    #      score = train_CNN(train_X, train_Y, dev_X, dev_Y, len(dataset[0]), 128)
    #     scores.append(score)
    
    # print ('Average score for {0}-fold cv: {1}'.format(args.folds, float(sum(scores)) / len(scores)))   


    ### ignore this
    # x shape 1166 / 216
    # x shape 1050 / 216 (cross val)

    # len dataset[0] = 216 + 1 (y)
    # len train_X[0] = 216
