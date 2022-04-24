import random
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# İndex relu. (This is the implementation of relu activation function we will use.)
def index_relu(x):
    x[x <= 0] = 0 
    return x

# index relu derivate (This is the derivative of our relu activation function.)
def index_relu_prime(m):
    m[m > 0] = 1
    m[m <= 0] = 0
    return m

# defining the metric as mse(mean squared error)
def mse(a, y):
    return 0.5* np.linalg.norm(a-y)**2  

# getting the data
data = pd.read_csv('../data.csv',sep=';')

#####################################################################################################################################################
# DATA PREPROCESSING

# seperating features and labels
X = data.iloc[:,1:].values
y = data['Price'].values

# encoding categorical data
one_hot = OneHotEncoder()
ct = ColumnTransformer(transformers=[('onehot',one_hot,[1,2,-1])],remainder='passthrough')
X = ct.fit_transform(X)

# splitting data into training and validation sets 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# scaling values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# combining features and labels
train_dataset = list(zip(X_train,y_train))
val_dataset = list(zip(X_valid,y_valid))


#####################################################################################################################################################

class Network(object):

    def __init__(self, sizes, seed=42):
        """Params:
            sizes: a list of which each element indicating the unit number for each layer 
        including input layer.
            seed: set seed to seed the generator."""

        # getting the number of layers
        self.num_layers = len(sizes)
        self.sizes = sizes

        # setting seed for randomized operations
        self.seed = seed
        random.seed(self.seed)

        # instantiating weights and biases according to the sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] 

    def predict(self, data):
        """Make prediction on a given data"""
        x,y = data
        x = np.expand_dims(x, axis=-1)
        out = self.feedforward(x)
        print(f"Prediction:{out}\nReal value:{y}")

    def feedforward(self, a):
        """We feed forward the data into the network and get the output."""
        for b, w in zip(self.biases, self.weights):
            a = index_relu(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lambda_, test_data=None, monitor_evaluation_cost=True, monitor_training_cost=True):
        """Train the neural network using mini -batch stochastic gradient
        descent. The ‘‘training_data ‘‘ is a list of tuples ‘‘(x, y)‘‘
        representing the training inputs and the desired outputs."""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, mini_batch_size, lambda_, n)
            
            print(f'EPOCH:{j}-----------------------------------------------------------------------------------')
            
            evaluation_cost = []
            training_cost = []

            if monitor_training_cost:
                cost = self.total_cost(training_data , lambda_)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))

            if monitor_evaluation_cost:
                cost = self.total_cost (test_data , lambda_)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            
     
    def update_mini_batch(self, mini_batch, eta, mini_batch_size, lambda_, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
     
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, lambda_, mini_batch, mini_batch_size)
            
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1- eta *(lambda_/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights , nabla_w)]
        self.biases = [b-( eta/len( mini_batch ))*nb for b, nb in zip(self.biases , nabla_b)]

    def backprop(self, x, y, lambda_, mini_batch, mini_batch_size):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        
        # feedforward
        x = np.expand_dims(x,axis=-1)
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = index_relu(z)
            activations.append(activation)         

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * index_relu_prime(np.array(zs[-1]))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 


        for l in range(2, self.num_layers):
            z = zs[-l]
            ip = index_relu_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * ip
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Evaluate the model with mse metric."""
        test_results = [(np.squeeze(self.feedforward(np.expand_dims(x,axis=-1)), axis=0) - y)**2 for x, y in test_data]
        return np.mean(test_results, axis=0)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def total_cost (self , data , lambda_):
        """Return the total cost for the data set ‘‘data ‘‘."""
        cost = 0.0
        for x, y in data:
            x = np.expand_dims(x, axis=-1)
            a = self. feedforward(x)
            cost += mse(a, y)/len(data)
            cost += 0.5*( lambda_/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

# Creating the model
model = Network(sizes=[11,128,64,64,1])

# Training the model
model.SGD(training_data=train_dataset, epochs=50, mini_batch_size=32, eta=0.000000001, lambda_=0, test_data=val_dataset)

# Predicting one of the validation data
model.predict(val_dataset[163])   


#####################################################################################################################################################
# 2nd PART: SAME MODEL WITH USING TENSORFLOW LIBRARY

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../data.csv',sep=';')
X = data.iloc[:,1:].values
y = data['Price'].values

one_hot = OneHotEncoder()
ct = ColumnTransformer(transformers=[('onehot',one_hot,[1,2,-1])],remainder='passthrough')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.fit_transform(X_valid)

train_dataset = list(zip(X_train,y_train))
val_dataset = list(zip(X_valid,y_valid))

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation = 'linear')
  ])

model.compile(loss=tf.keras.losses.mean_absolute_error,
              optimizer =tf.keras.optimizers.SGD(learning_rate=0.05),
              metrics = ['mse'])

model.fit(X_train,
          y_train,
          epochs=50,
          validation_data=(X_valid,y_valid)) 
