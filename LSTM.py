# Os modelos LSTM recaem na classe de inteligência Artificial conhecida como Deep Learning. Sua aplicação fica muito mais fácil 
# quando usamos o Python 3 com o pacote Keras e o tensorflow, que é o que iremos mostrar aqui. 
# O exemplo abaixo faz a previsão do S&P500
###
###
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#####
def load_data(filename, seq_len, normalise_window,train=(0.9)):
    # filename: ex: "name.csv". Doesn`t use timestamp, just the univariate column
    # seq_len: window size = steps to forecast
    # normalise_window: window to normalise. There`s no default
    # train: percent of train
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')  
    #
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    #
    if normalise_window:
        result = normalise_windows(result)   #normalise
    #
    result = np.array(result)   #convert to numpy
    #
    row = round(train * result.shape[0])   #create 90% to train
    train = result[:int(row), :]   #create train
    ## Change the order of the train (maybe is not good for time series)
    np.random.shuffle(train)     
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    # reshape to (N,W,F)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]
#####
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data
####
def normalise(dataset):
    scaler=MinMaxScaler(feature_range=(0,1))
    df=scaler.fit_transform(dataset)
    return df, scaler
###
### Creating model LSTM
### [1,L1,L2,L3,1] 1 input, 3 layers [L1,L2,L3], 1 output
def build_model(layers):
    model=Sequential()
    model.add(LSTM(input_dim=layers[0],
                  output_dim=layers[1],
                  return_sequences=True))
    model.add(LSTM(layers[2],
                  return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(layers[3],
                  return_sequences=False))
    model.add(Dense(output_dim=layers[4]))
    model.add(Activation("linear"))
    start=time.time()
    model.compile(loss="mse",optimizer="rmsprop")
    print("> Compilation time: ",time.time()-start)
    return model
###
def predict_point_by_point(model,data):
    predicted=model.predict(data)
    predicted=np.reshape(predicted,(predicted.size,))
    return predicted
###
def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted
###
def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
###
## Plots
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
##
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        #plt.legend()
    plt.show()
 ########
 ### Run model
epochs=200
seq_len=50
x_train,y_train,x_test,y_test=load_data("sp500.csv",seq_len,True)
model1=build_model([1,100,50,25,1])
model1.fit(x_train,y_train,batch_size=512,nb_epoch=epochs,validation_split=0.05)
predicted=predict_point_by_point(model1,x_test)
predictions=predict_sequences_multiple(model1,x_test,seq_len,50)
#
plot_results(predicted,y_test)   #normalised data
#
plot_results_multiple(predictions,y_test,50)
##############
