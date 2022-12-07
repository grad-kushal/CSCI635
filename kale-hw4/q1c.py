import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# meta-parameters for program
prob = "Q1C_"
beta = 0.01  # regularization coefficient or lambda
alpha = 0.01  # step size coefficient or learning rate
eps = 0.000001  # controls convergence criterion
n_epoch = 250000  # number of epochs (full passes through the dataset)
trial_name = prob + "beta:" + str(beta) + "alpha:" + str(
    alpha) + "n_epochs:" + str(n_epoch)  # will add a unique sub-string to output of this program
epsilon = 0.001  # secant approximation


def load_data(file_name):
    """
    Loads the dataset
    :return: The dataset
    """
    path = os.getcwd() + file_name
    data = pd.read_csv(path, header=None)
    return data


def one_hot_encoding(y):
    """
    One hot encodes the labels
    :param y: Labels
    :return: One hot encoded labels
    """
    y = y.reshape(-1)
    max = np.max(y) + 1
    y = np.eye(max)[y]
    return y


def rectified_linear(z):
    """
    Computes the rectified linear unit function
    :param z: Input
    :return: Rectified linear unit function output
    """
    return np.maximum(z, 0)


def sigmoid(x):
    """
    Computes the sigmoid function
    :param x: Input
    :return: Sigmoid function output
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Computes the tanh function
    :param x: Input
    :return: Tanh function output
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x):
    """
    Computes the softmax function
    :param x: Input
    :return: Softmax function output
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def forward_propagation(X, theta):
    """
    Computes the forward propagation
    :param X: Dataset
    :param theta: Model parameters
    :return: Forward propagation output
    """
    W, c, w, b = theta
    y1 = np.dot(X, W.T) + c
    activated1 = tanh(y1)
    y2 = np.dot(activated1, w.T) + b
    activated2 = softmax(y2)
    return activated1, y1, activated2, y2


def compute_cost(X, y, theta):
    """
    Computes the cost function
    :param X: Dataset
    :param y: Labels
    :param theta: Model parameters
    :return: Cost function output
    """
    m = X.shape[0]  # number of training examples
    activated1, y1, activated2, y2 = forward_propagation(X, theta)
    # L = -(1 / m) * np.sum(y * np.log(activated2) + (1 - y) * np.log(1 - activated2))
    a = np.log(activated2)
    b = np.log(1 - activated2)
    one_minus_y = 1 - y
    L = -(1 / m) * np.sum(y * a + one_minus_y * b)
    R = (beta / 2) * (np.sum(np.square(theta[0])) + np.sum(np.square(theta[2])))
    L += R
    return L, activated1, y1, activated2, y2


def compute_grad(m, X, y, theta, activated1, y1, activated2):
    """
    Computes the gradient
    :param X: Dataset
    :param m: Number of training examples
    :param y: Labels
    :param theta: Model parameters
    :param activated1: Output of the first layer
    :param y1: Output of the first layer before activation
    :param activated2: Output of the second layer
    :return: Gradient
    """
    W, c, w, b = theta
    dy2 = activated2 - y
    db = (1 / m) * np.sum(dy2, axis=0)
    dw = (1 / m) * np.dot(dy2.T, activated1)
    dw = beta * w + dw
    dy1 = np.dot(dy2, w.T) * sigmoid(y1) * (1 - sigmoid(y1))
    dc = (1 / m) * np.sum(dy1, axis=0)
    dW = (1 / m) * np.dot(dy1.T, X)
    dW += beta * W
    return dW, dc, dw, db


def predict(X, theta):
    """
    Predicts the labels
    :param X: Dataset
    :param theta: Model parameters
    :return: Predicted labels
    """
    _, _, activated2, _ = forward_propagation(X, theta)
    return np.argmax(activated2, axis=1)


def check_grad(dW, dc, dw, db, X, one_hot_encoding_y, theta):
    """
    Compares the computed gradient with the numerical gradient using SECANT APPROXIMATION
    :param theta: Model parameters
    :param dW: Computed gradient of the first layer
    :param dc: Computed gradient of the first layer bias
    :param dw: Computed gradient of the second layer
    :param db: Computed gradient of the second layer bias
    :param X: Dataset
    :param one_hot_encoding_y: One-hot encoded labels
    :return: Difference between the computed gradient and the numerical gradient
    """
    dW = dW.flatten()
    dc = dc.flatten()
    dw = dw.flatten()
    db = db.flatten()
    grad = np.concatenate((dW, dc, dw, db))
    numgrad = np.zeros(grad.shape)
    perturb = np.zeros(grad.shape)
    for i in range(len(grad)):
        perturb[i] = epsilon
        theta_temp = (theta[0] - perturb[:12].reshape(3, 4), theta[1] - perturb[12:15].reshape(1, 3),
                      theta[2] - perturb[15:24].reshape(3, 3), theta[3] - perturb[24:].reshape(1, 3))
        L1, _, _, _, _ = compute_cost(X, one_hot_encoding_y, theta_temp)
        theta_temp = (theta[0] + perturb[:12].reshape(3, 4), theta[1] + perturb[12:15].reshape(1, 3),
                      theta[2] + perturb[15:24].reshape(3, 3), theta[3] + perturb[24:].reshape(1, 3))
        L2, _, _, _, _ = compute_cost(X, one_hot_encoding_y, theta_temp)
        numgrad[i] = (L2 - L1) / (2 * epsilon)
        perturb[i] = 0
    return np.abs(np.subtract(numgrad, grad)).min()


def main():
    """
    Main function
    :return: None
    """
    # Load the dataset
    data = load_data('/iris_train.dat')
    cols = data.shape[1]
    # set X (training data) and y (target variable)
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    # convert from data frames to numpy matrices
    X = np.array(X.values)
    y = np.array(y.values)
    one_hot_encoding_y = one_hot_encoding(y)
    print("X.shape = " + str(X.shape))
    print("y.shape = " + str(y.shape))
    print("one_hot_encoding_y.shape = " + str(one_hot_encoding_y.shape))

    np.random.seed(654)  # Random seed to get consistent results
    m = X.shape[0]  # number of training examples
    W = np.random.randn(3, 4)  # initialize W randomly
    c = np.random.randn(1, 3)  # initialize c randomly
    w = np.random.randn(1, 3)  # initialize w randomly
    b = np.random.randn(1, 1)  # initialize b randomly
    theta = (W, c, w, b)

    cost = []
    epochs = []

    np.set_printoptions(suppress=True)

    # Run the algorithm
    for epoch in range(n_epoch):
        epochs.append(epoch)
        L, activated1, y1, activated2, y2 = compute_cost(X, one_hot_encoding_y, theta)
        if epoch % 10000 == 1:
            print("Epoch: " + str(epoch) + " Loss: " + str(L))
        cost.append(L)
        # Backward propagation
        dW, dc, dw, db = compute_grad(m, X, one_hot_encoding_y, theta, activated1, y1, activated2)
        diff = check_grad(dW, dc, dw, db, X, one_hot_encoding_y, theta)
        # diff = 0.00000001
        # print("Diff: " + str(diff))
        if diff > 1e-3:
            print("Gradient check failed")
            break
        else:
            if epoch % 10000 == 2:
                print("Gradient check passed")
            # Gradient descent parameter update
            W = W - alpha * dW
            c = c - alpha * dc
            w = w - alpha * dw
            b = b - alpha * db
            theta = (W, c, w, b)

    plt.plot(epochs, cost)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.savefig("out/" + trial_name + '_cost_vs_epoch.png')
    plt.show()

    output = predict(X, theta)
    print("output: " + str(output))
    error = 0
    for i in range(len(output)):
        if output[i] != y[i]:
            error += 1
    error /= len(y)
    print("Train Accuracy = ", round(100 * (1 - error)), "%")

    plt.scatter(X[:, 0], X[:, 1], c=output)
    plt.savefig("out/" + trial_name + 'scatter.png')
    plt.show()

    ####################################################
    data = load_data('/iris_test.dat')
    cols = data.shape[1]
    # set X (training data) and y (target variable)
    Xt = data.iloc[:, 0:cols - 1]
    yt = data.iloc[:, cols - 1:cols]

    # convert from data frames to numpy matrices
    Xt = np.array(Xt.values)
    yt = np.array(yt.values)
    one_hot_encoding_y = one_hot_encoding(y)

    output = predict(Xt, theta)
    error = 0
    for i in range(len(output)):
        if output[i] != yt[i]:
            error += 1
    error /= len(yt)
    print("Test Accuracy = ", round(100 * (1 - error)), "%")

    plt.scatter(Xt[:, 0], Xt[:, 1], c=output)
    plt.savefig("out/" + trial_name + 'scatter_test.png')
    plt.show()


if __name__ == '__main__':
    main()
