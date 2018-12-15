# Esse notebook ilustra como usar o keras para prever o comportamento do Bitcoin (BTC). 
# O pacote keras, que pode ser acessado a partir do Python, é um dos mais amplos e fáceis de usar para aplicações que
# envolvem inteligência artificial.
###########
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
##
diretorio=os.getcwd()
data_csv=pd.read_csv(diretorio+'/'+'btc.csv',index_col="datetime")
print(data_csv.shape)
data_csv.head(2)
##
data_to_use=100
train_end=70
total_data=int(len(data_csv))
start=total_data-data_to_use
## 
steps_to_predict=3
train_mse=[]
test_mse=[]
forecast=[]
for i in range(steps_to_predict):
    train_mse.append(0)
    test_mse.append(0)
    forecast.append(0)
##
yt=data_csv.iloc[start:total_data,3]   #close
yt1=data_csv.iloc[start:total_data,2]  #open
yt2=data_csv.iloc[start:total_data,1]  #high
yt3=data_csv.iloc[start:total_data,0]  #low
vt=data_csv.iloc[start:total_data,5]  #volume
#####
for i in range(steps_to_predict):
    if i==0:
        units=20
        batch_size=1
    if i==1:
        units=15
        batch_size=1
    if i==2:
        units=80
        batch_size=1
    yt_=yt.shift(-i-1)
    datat=pd.concat([yt,yt_,vt,yt1,yt2,yt3],axis=1)
    datat.columns=['yt','yt_','vt','yt1','yt2','yt3']
    datat=datat.dropna()
    y=datat["yt_"]   #target variable - closed price
    cols=["yt","vt","yt1","yt2","yt3"]
    x=datat[cols]
    scaler_x=preprocessing.MinMaxScaler(feature_range=(-1,1))
    x=np.array(x).reshape((len(x),len(cols)))
    x=scaler_x.fit_transform(x)
    scaler_y=preprocessing.MinMaxScaler(feature_range=(-1,1))
    y=np.array(y).reshape((len(y),1))
    y=scaler_y.fit_transform(y)
    x_train=x[0:train_end,]
    x_test=x[train_end+1:len(x),]
    y_train=y[0:train_end]
    y_test=y[train_end+1:len(y)]
    if(i==0):
        prediction_data=[]
    for j in range(len(y_test)-0):
        prediction_data.append(0)
    ###
    x_train=x_train.reshape(x_train.shape+(1,))
    x_test=x_test.reshape(x_test.shape+(1,))
    ###
    seed=2018
    np.random.seed(seed)
##### i=0
    if i==0:
        fit0=Sequential()
        fit0.add(LSTM(units,activation="tanh",inner_activation="hard_sigmoid",input_shape=(len(cols),1)))
        fit0.add(Dropout(0.2))
        fit0.add(Dense(output_dim=1,activation="linear"))
        fit0.compile(loss="mean_squared_error",optimizer="adam")
        fit0.fit(x_train,y_train,batch_size=batch_size,nb_epoch=25,shuffle=False)
        train_mse[i]=fit0.evaluate(x_train,y_train,batch_size=batch_size)
        test_mse[i]=fit0.evaluate(x_test,y_test,batch_size=batch_size)
        pred=fit0.predict(x_test)
        pred=scaler_y.inverse_transform(np.array(pred).reshape((len(pred),1)))
        for j in range(len(pred)-0):
            prediction_data[j]=pred[j]
    
        forecast[i]=pred[-1]
##### i=1
    if i==1:
        fit1=Sequential()
        fit1.add(LSTM(units,activation="tanh",inner_activation="hard_sigmoid",input_shape=(len(cols),1)))
        fit1.add(Dropout(0.2))
        fit1.add(Dense(output_dim=1,activation="linear"))
        fit1.compile(loss="mean_squared_error",optimizer="adam")
        fit1.fit(x_train,y_train,batch_size=batch_size,nb_epoch=25,shuffle=False)
        train_mse[i]=fit1.evaluate(x_train,y_train,batch_size=batch_size)
        test_mse[i]=fit1.evaluate(x_test,y_test,batch_size=batch_size)
        pred=fit1.predict(x_test)
        pred=scaler_y.inverse_transform(np.array(pred).reshape((len(pred),1)))
        forecast[i]=pred[-1]
##### i=2
    if i==2:
        fit2=Sequential()
        fit2.add(LSTM(units,activation="tanh",inner_activation="hard_sigmoid",input_shape=(len(cols),1)))
        fit2.add(Dropout(0.2))
        fit2.add(Dense(output_dim=1,activation="linear"))
        fit2.compile(loss="mean_squared_error",optimizer="adam")
        fit2.fit(x_train,y_train,batch_size=batch_size,nb_epoch=250,shuffle=False)
        train_mse[i]=fit2.evaluate(x_train,y_train,batch_size=batch_size)
        test_mse[i]=fit2.evaluate(x_test,y_test,batch_size=batch_size)
        pred=fit2.predict(x_test)
        pred=scaler_y.inverse_transform(np.array(pred).reshape((len(pred),1)))
        forecast[i]=pred[-1]
    
    x_test=scaler_x.inverse_transform(np.array(x_test).reshape((len(x_test),len(cols))))
###
prediction_data=np.asarray(prediction_data)
prediction_data=prediction_data.ravel()
###
for j in range(len(prediction_data)-1):
    prediction_data[len(prediction_data)-j-1]=prediction_data[len(prediction_data)-1-j-1]

prediction_data=np.append(prediction_data,forecast)
x_test_all=yt[len(yt)-len(prediction_data)-1:len(yt)-1]
x_test_all=x_test_all.ravel()
##
plt.plot(prediction_data,label="predictions")
plt.plot(x_test_all,label="actual")
plt.legend(loc="best",shadow=True,fancybox=True)
import matplotlib.ticker as mtick
fmt="$%.0f"
tick=mtick.FormatStrFormatter(fmt)
ax=plt.axes()
ax.yaxis.set_major_formatter(tick)
plt.show()
#####
