# Esse notebook implementa inteligência artificial para prever o preço de um ativo.
# Usamos como exemplo o comportamento do Bitcoin (BTC).
# A rede neural construída considera uma estrutura do tipo Sequential() contida no keras e python 3
# Como insumo, usamos diversos indicadores técnicos aplicados ao OHLC do BTC.
# Após termos o modelo, testamos o mesmo para ver sua eficácia calculando o retorno acumulado
##
##
import pandas as pd
import numpy as np
import talib as ta
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.signal import argrelextrema
import json
from bs4 import BeautifulSoup
import requests 
import urllib.request
import datetime
from datetime import timedelta, datetime
import time
#
#
dia=time.strftime("%d%m%Y")
diretorio= os.getcwd()
moed=["BTC","ETH","BCH"]
for coin in moed:
    if coin=="MIOTA":
        coin="IOTA"
    urlm = "https://min-api.cryptocompare.com/data/histoday?fsym="+str(coin)+"&tsym=USD&limit=2000&aggregate=1&e=CCCAGG"  #limit control the total of currency
    resp = requests.get(urlm)
    soupe = BeautifulSoup(resp.content, "html.parser")
    dictt = json.loads(soupe.prettify())
    dic5=dictt["Data"]
    dad5 = pd.DataFrame(columns=["time","open","high","low","close","volume"])             
    for i in range(len(dic5)):
        dad5.loc[len(dad5)] = [dic5[i]['time'],dic5[i]['open'],dic5[i]["high"],dic5[i]["low"],
                         dic5[i]["close"],dic5[i]["volumeto"]]
    dad5['datetime'] = pd.to_datetime(dad5.time, unit='s')
    dad5=dad5.drop(['time'],axis=1)
    dad5=dad5.set_index('datetime')
    # Salva para analise futura
    dad5.to_csv(diretorio+"/"+str(coin)+str(dia)+'.csv',sep=";")
#
#
diretorio=os.getcwd()
dataset=pd.read_csv(diretorio+"/"+"BTC"+str(dia)+".csv",sep=";",index_col="datetime")
dataset.tail(2)
#
#
# indicadores
dataset["h-l"]=dataset["high"]-dataset["low"]
dataset["o-c"]=dataset["close"]-dataset["open"]
dataset["3day_ma"]=dataset["close"].shift(1).rolling(window=3).mean()
dataset["10day_ma"]=dataset["close"].shift(1).rolling(window=10).mean()
dataset["30day_ma"]=dataset["close"].shift(1).rolling(window=30).mean()
dataset['ema1'] = ta.EMA(dataset.close,timeperiod = 5)  #exponential moving average with time_ema1
dataset['ema2'] = ta.EMA(dataset.close,timeperiod = 18)  #exponential moving average with time_ema2
dataset['macd'],dataset['macdsignal'],dataset['macdhist']=ta.MACD(dataset.close,
                                                                  fastperiod=12,slowperiod=26,signalperiod=9) 
dataset['mom']=ta.MOM(dataset.close, timeperiod=6) 
dataset["apo"]=ta.APO(dataset.close,fastperiod=6,slowperiod=18,matype=0)
dataset["rsi"]=ta.RSI(dataset["close"].values,timeperiod=9)
dataset['upperband'],dataset['middleband'],dataset['lowerband']=ta.BBANDS(dataset.close,
                                                                          timeperiod=5,nbdevup=2,nbdevdn=2,matype=0)
dataset["williams"]=ta.WILLR(dataset["high"].values,
                             dataset["low"].values,
                             dataset["close"].values,7)
#
# Vamos criar sinais de compra e venda com base em indicadores tecnicos usados no mercado
## RSI   
dataset['rsi_s']=0
dataset.loc[(dataset['rsi'] >= 60),'rsi_s']=-1
dataset.loc[(dataset['rsi'] <= 40),'rsi_s']=1
## MACDhist
dataset["macd_s"]=0
dataset.loc[dataset["macdhist"]>dataset["macdhist"].shift(2),'macd_s']=1
dataset.loc[dataset["macdhist"]<dataset["macdhist"].shift(2),'macd_s']=-1
## turning point close
dataset['minimo'] = dataset.iloc[argrelextrema(dataset.close.values, np.less_equal, order=3)[0]]['close']
dataset['maximo'] = dataset.iloc[argrelextrema(dataset.close.values, np.greater_equal, order=3)[0]]['close']
dataset.minimo.fillna(0,inplace=True)
dataset.maximo.fillna(0,inplace=True)
dataset['turning_point']=0
dataset.loc[(dataset["minimo"]>0),'turning_point']=1
dataset.loc[(dataset["maximo"]>0),'turning_point']=-1
## turning point bolling bands
dataset['bb_minimo'] = dataset.iloc[argrelextrema(dataset.close.values, np.less_equal, order=3)[0]]['lowerband']
dataset['bb_maximo'] = dataset.iloc[argrelextrema(dataset.close.values, np.greater_equal, order=3)[0]]['upperband']
dataset.bb_minimo.fillna(0,inplace=True)
dataset.bb_maximo.fillna(0,inplace=True)
dataset['tp_bb']=0
dataset.loc[(dataset["bb_minimo"]>0),'tp_bb']=1
dataset.loc[(dataset["bb_maximo"]>0),'tp_bb']=-1
## turning point MOM
dataset["mom_min"]=dataset.iloc[argrelextrema(dataset.mom.values,np.less_equal,order=10)[0]]['mom']
dataset["mom_max"]=dataset.iloc[argrelextrema(dataset.mom.values,np.greater_equal,order=10)[0]]["mom"] 
dataset.mom_min.fillna(0,inplace=True)
dataset.mom_max.fillna(0,inplace=True)
dataset['tp_mom']=0
dataset.loc[(dataset["mom_min"]>0),'tp_mom']=1
dataset.loc[(dataset["mom_max"]>0),'tp_mom']=-1
## turning point APO
dataset["apo_min"]=dataset.iloc[argrelextrema(dataset.apo.values,np.less_equal,order=8)[0]]['apo']
dataset["apo_max"]=dataset.iloc[argrelextrema(dataset.apo.values,np.greater_equal,order=8)[0]]["apo"] 
dataset.apo_min.fillna(0,inplace=True)
dataset.apo_max.fillna(0,inplace=True)
dataset['tp_apo']=0
dataset.loc[(dataset["apo_min"]>0),'tp_apo']=1
dataset.loc[(dataset["apo_max"]>0),'tp_apo']=-1
## candle
dataset['candle']=0
dataset.loc[(dataset['close']>dataset['ema2'])&(dataset['close'].shift(-1)<dataset['ema2'].shift(-1)),'candle']=-1
dataset.loc[(dataset['close']<dataset['ema2'])&(dataset['close'].shift(-1)>dataset['ema2'].shift(-1)),'candle']=1
#
dataset=dataset.drop(["open","high","low","minimo","maximo","mom_min","mom_max","bb_minimo","bb_maximo",
                     "apo_min","apo_max"],axis=1)
#                    
#
# y variable
dataset["y"]=np.where(dataset["close"].shift(-1)>dataset["close"],1,0)    #valor 1=alta
dataset=dataset.dropna()
print(dataset.shape)
dataset.tail(4)
#
x=dataset.iloc[:,:-1]
x.columns.values.tolist()
#
y=pd.DataFrame(dataset.iloc[:,-1])
## train and test
split = int(len(dataset)*0.8)
x_train, x_test, y_train, y_test = x[:split],x[split:],y[:split],y[split:]
#
### scaling. Veja anexo para uma discussao sobre qual o melhor scaler
from sklearn.preprocessing import StandardScaler, QuantileTransformer
# standardscaler
#scaler1=StandardScaler()
#x_train=scaler1.fit_transform(x_train)
#x_test=scaler1.transform(x_test)
# quantiletransformer
scaler2=QuantileTransformer(output_distribution='uniform')
x_train=scaler2.fit_transform(x_train)
x_test=scaler2.fit_transform(x_test)
#
## ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout
classifier=Sequential()
classifier.add(Dense(units=512,kernel_initializer="uniform",activation="sigmoid",input_dim=x.shape[1]))
classifier.add(Dense(units=256,kernel_initializer="uniform",activation="sigmoid"))
classifier.add(Dense(units=128,kernel_initializer="uniform",activation="sigmoid"))
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))   #output layer
classifier.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"]);
#
### applying the model
classifier.fit(x_train,y_train,batch_size=80,epochs=200,verbose=2)
#
### predicting
y_pred=classifier.predict(x_test)   # tem valores entre 0 e 1 que representam probabilidades
y_pred2=(y_pred>0.7)  #cria uma coluna y_pred>0.5 entao temos true (chance do mercado subir), o contrario false.
print(y_pred2.shape)
plt.plot(y_pred)
#
dataset["y_pred2"]=np.NaN
dataset.iloc[(len(dataset)-len(y_pred)):,-1:]=y_pred2   #armazena os valores apenas na parte de teste do dataset
trade_dataset=dataset.dropna()
trade_dataset["y_pred"]=y_pred;
print(trade_dataset.shape)
trade_dataset.tail(3)
#
### Strategy return
trade_dataset["tomorrow_ret"]=0.
trade_dataset["tomorrow_ret"]=np.log(trade_dataset["close"]/trade_dataset["close"].shift(1))
trade_dataset["tomorrow_ret"]=trade_dataset["tomorrow_ret"].shift(-1)
#
trade_dataset["strategy_ret"]=0.
trade_dataset["strategy_ret"]=np.where(trade_dataset["y_pred2"]==True,trade_dataset["tomorrow_ret"],0)
#
# ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. 
# This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, 
# and a true positive rate of one. This is not very realistic, but it does mean that a larger 
# area under the curve (AUC) is usually better.
from sklearn.metrics import roc_curve, roc_auc_score, auc
fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(2):   #duas classes
    fpr[i],tpr[i], _ = roc_curve(trade_dataset.y,trade_dataset.y_pred)
    roc_auc[i] = auc(fpr[i],tpr[i])   #calcula a area
#
fpr["micro"],tpr["micro"], _ = roc_curve(trade_dataset.y,trade_dataset.y_pred)
roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])    
print("ROC area: ",roc_auc_score(trade_dataset.y,trade_dataset.y_pred))
#
plt.plot(fpr[1],tpr[1],color="darkorange",lw=2,label="ROC area= %0.3f" % roc_auc[1])
plt.plot([0,1],[0,1],color="navy",lw=1,linestyle="--")
plt.legend(loc="best")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate");
#
# cumulative returns
n_dias=100
investing=trade_dataset[-n_dias:]
investing["cum_market_ret"]=np.cumsum(investing["tomorrow_ret"])
investing["cum_strategy_ret"]=np.cumsum(investing["strategy_ret"]);
##
print("Retorno dos últimos ",n_dias,"  dias")
print("Market Return Cumulative em %",investing["cum_market_ret"][-2]*100)
print("Strategy Return Cumulative em %",investing["cum_strategy_ret"][-2]*100)
plt.figure(figsize=(10,5))
plt.plot(investing["cum_market_ret"],color="r",label="Market Return")
plt.plot(investing["cum_strategy_ret"],color="b",label="Strategy Return")
plt.legend(loc="best")
plt.show()
#
##############
