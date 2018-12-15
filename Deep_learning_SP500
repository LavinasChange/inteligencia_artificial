# Esse notebook ilustra como podemos fazer a previsao do preco do SP500 as partir do preco de 500 outras acoes que compoem 
# o indice usando deep learning com tensorflow. Destaque que aqui usamos uma rede neural do tipo feedforward.
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
diretorio=os.getcwd()
#
dat=pd.read_csv(diretorio+'/'+'data_stocks.csv')    #wide format
dat.head(3)
#
# O valor do SP500 corresponde a t+1 enquanto que as demais acoes estao em t. Isso é feito pois 
# o objetivo é fazer a previsao para o momento seguinte. Assim, usamos o preco das acoes em t (minuto atual) 
# para prever o valor do indice em t+1 (minuto seguinte). Para esse exercicio eliminamos as datas e convertemos os 
# dados para o formato numpy, usado no tensorflow.
#
data=dat.drop(["DATE"],1)
n,p=data.shape[0],data.shape[1]
data=data.values
plt.plot(dat["SP500"]);
#
# Vamos dividir o conjunto de dados entre treino e teste
split=0.8   #80%
train_start=0
train_end=int(np.floor(split*n))    #np.floor return the max integer
test_start=train_end
test_end=n
data_train=data[np.arange(train_start,train_end),:]
data_test=data[np.arange(test_start,test_end),:]
#
# Usamos um "scalar" para normalizar os dados
scaler=MinMaxScaler()
scaler.fit(data_train)   #usa o scaler no conjunto de dados de treino e depois aplica para o teste.
data_train, data_test =scaler.transform(data_train),scaler.transform(data_test)
x_train=data_train[:,1:]   #skip the first column
y_train=data_train[:,0]   #y is the first column (sp500 index)
x_test=data_test[:,1:]
y_test=data_test[:,0]
#
# Montando a arquitetura da rede
# Placeholder = Usado para armazenar os inputs e o target.
# Seja um modelo com 4 layers e a seguinte quantidade de neuronios (1024,512,256,128)
n_stocks=500
n_neurons_1=1024
n_neurons_2=512
n_neurons_3=256
n_neurons_4=128
n_target=1
X=tf.placeholder(dtype=tf.float32,shape=[None,n_stocks])
Y=tf.placeholder(dtype=tf.float32,shape=[None])
#
# initializer (ha diversos que podem ser escolhidos)
sigma=1
weight_initializer=tf.variance_scaling_initializer(mode="fan_avg",distribution="uniform",scale=sigma)
bias_initializer=tf.zeros_initializer()
#
# Layer 1
W_hidden_1 = tf.Variable(weight_initializer([n_stocks,n_neurons_1]))
bias_hidden_1=tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1,n_neurons_2]))
bias_hidden_2=tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2,n_neurons_3]))
bias_hidden_3=tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3,n_neurons_4]))
bias_hidden_4=tf.Variable(bias_initializer([n_neurons_4]))
# output
W_out = tf.Variable(weight_initializer([n_neurons_4,n_target]))
bias_out=tf.Variable(bias_initializer([n_target]))
#
# Activation function   (pode escolher dentre varias)
# Hidden Layer
hidden_1=tf.nn.relu(tf.add(tf.matmul(X,W_hidden_1),bias_hidden_1))
hidden_2=tf.nn.relu(tf.add(tf.matmul(W_hidden_1,W_hidden_2),bias_hidden_2))
hidden_3=tf.nn.relu(tf.add(tf.matmul(W_hidden_2,W_hidden_3),bias_hidden_3))
hidden_4=tf.nn.relu(tf.add(tf.matmul(W_hidden_3,W_hidden_4),bias_hidden_4))
out=tf.nn.relu(tf.add(tf.matmul(W_hidden_4,W_out),bias_out))
#
# Definindo a funcao custo e o optimizer
mse=tf.reduce_mean(tf.squared_difference(out,Y))   #ha varias funcoes custo
opt=tf.train.AdamOptimizer().minimize(mse)   # ha varias funcoes de otimizacao
#
# Treinando o modelo
# Make session
net=tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())
#
epochs=10
batch_size=256
for e in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: x_test, Y: y_test})
print(mse_final)
# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1=fig.add_subplot(111)
line1,=ax1.plot(y_test)
plt.show()
#
