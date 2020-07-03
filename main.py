import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import scipy


data = pd.read_csv("dataset.csv")

#print(data.head)

#data.info()


# Preparing text data from DATASET 1
#data.classes = [1 if each == "2" else 0 for each in data.classes]
for i in range(0,len(data.classes)):
    data.classes[i] = int(data.classes[i]/4)
#print(data.classes)
y = data.classes.values
#print(y.shape)
x_data = data.drop(['classes'], axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

x_train1 = x_train.T
x_test1 = x_test.T
y_train1 = y_train.T
y_test1 = y_test.T

#print(x)
#print(y)
#print(y_train1)
#print(y_test1)

# Preparing the image data from DATASET 2

ROWS = 64
COLS = 64
CHANNELS = 3
TRAIN_DIR = 'training_set/training_set/'
TEST_DIR = 'test_set/test_set/'
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
def prepare_image_data(images):
    m = len(images)
    X = np.zeros((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
    y = np.zeros((1, m))
    for i, image_file in enumerate(images):
        X[i,:] = read_image(image_file)
        if 'dog' in image_file.lower():
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0
    return X, y



def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize(input_layer, hidden_layer, output_layer):
    W1 = np.random.randn(hidden_layer, input_layer) * 0.01
    b1 = np.zeros((hidden_layer, 1))
    W2 = np.random.randn(output_layer, hidden_layer) * 0.01
    b2 = np.zeros((output_layer, 1))
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters

def forward(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    dict = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
    return A2, dict

def cost(A2, Y, parameters, key):
    if key == 1:
        m = Y.shape[1]
    else:
        m = Y.shape[0]
    #print(m)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -1 / m * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def backward(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    gd = {"dW1": dW1,"db1": db1,"dW2": dW2,"db2": db2}
    return gd

def update(parameters, grads, learning_rate=0.1):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters

def predict(parameters, X):
    Y_prediction = np.zeros((1, X.shape[1]))
    A2, cache = forward(X, parameters)

    for i in range(A2.shape[1]):
        if A2[0, i] > 0.5:
            Y_prediction[[0], [i]] = 1
        else:
            Y_prediction[[0], [i]] = 0

    return Y_prediction

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b


def model(X_train, Y_train, X_test, Y_test, n_h, num_iterations=1000, learning_rate=0.05,key=0, print_cost=False):
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    parameters = initialize(n_x, n_h, n_y)

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    costs = []
    for i in range(0, num_iterations):
        A2, cache = forward(X_train, parameters)
        cst = cost(A2, Y_train, parameters, key)
        grads = backward(parameters, cache, X_train, Y_train)
        parameters = update(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cst))
        if i % 100 == 0:
            costs.append(cst)
    Y_prediction_test = predict(parameters, X_test)
    Y_prediction_train = predict(parameters, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    parameters.update({"costs": costs, "n_h": n_h,"learning_rate": learning_rate})
    return parameters

d = model(x_train1, y_train1, x_test1, y_test1,1, num_iterations = 1000, learning_rate = 0.01, print_cost = True)
'''
train_set_x, train_set_y = prepare_image_data(train_images)
test_set_x, test_set_y = prepare_image_data(test_images)

train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], ROWS*COLS*CHANNELS).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#d = model(train_set_x, train_set_y, test_set_x, test_set_y,1, num_iterations = 1000, learning_rate = 0.001,key=1, print_cost = True)

#running model on both datasets

print("############### Running Logistic Regression on DATASET 1 ###############")
learning_rates = [0.01,0.001,0.0001,0.1,0.5]
models = {}
for i in learning_rates:
    print ("learning rate is: ",i)
    models[i] = model(x_train1, y_train1, x_test1, y_test1,1, num_iterations = 1000, learning_rate = i,key=0, print_cost = True)
    print ("-------------------------------------------------------")
for i in learning_rates:
    plt.plot(np.squeeze(models[i]["costs"]), label= str(models[i]["learning_rate"]))
plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

print("############### End of DATASET 1 Results ###############")

print("############### Running Logistic Regression on DATASET 2 ###############")

learning_rates = [0.01,0.001,0.0001,0.1,0.5]
models = {}
for i in learning_rates:
    print ("learning rate is: ",i)
    models[i] = model(train_set_x, train_set_y, test_set_x, test_set_y,1, num_iterations = 1000, learning_rate = i,key=1, print_cost = True)
    print ("-------------------------------------------------------")
for i in learning_rates:
    plt.plot(np.squeeze(models[i]["costs"]), label= str(models[i]["learning_rate"]))
plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
'''