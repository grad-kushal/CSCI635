import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# meta-parameters for program
prob = "Q1A_"
beta = 0.001  # regularization coefficient
alpha = 0.0005  # step size coefficient
eps = 0.000001  # controls convergence criterion
n_epoch = 50000  # number of epochs (full passes through the dataset)
trial_name = prob + "beta:" + str(beta) + "alpha:" + str(
    alpha) + "n_epochs:" + str(n_epoch)  # will add a unique sub-string to output of this program
epsilon = 0.001  # secant approximation


def load_data():
    """
    Loads the dataset
    :return: The dataset
    """
    path = os.getcwd() + '/xor.dat'
    data = pd.read_csv(path, header=None)
    return data


def onehotencoding(y):
    """
    One-hot encodes the labels
    :param y: Labels
    :return: One-hot encoded labels
    """
    y = y.reshape(-1)
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


def rectified_linear(z):
    """
    Computes the rectified linear unit function
    :param x: Input
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


def forward_propagation(X, theta):
    """
    Computes the forward propagation of the model
    :param X: Dataset
    :param theta: Model parameters
    :return: Forward propagation output
    """
    W, c, w, b = theta
    y1 = np.dot(X, W.T) + c
    activated1 = sigmoid(y1)
    y2 = np.dot(activated1, w.T) + b
    activated2 = rectified_linear(y2)
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
    L = -(1 / m) * np.sum(y * np.log(activated2) + (1 - y) * np.log(1 - activated2))
    R = (beta / 2) * (np.sum(np.square(theta[0])) + np.sum(np.square(theta[2])))
    L += R
    return L, activated1, y1, activated2, y2


def compute_grad(m, X, y, theta, activated1, y1, activated2):
    """
    Computes the gradient of the cost function
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
    dy1 = np.dot(dy2, w.T) * activated1 * (1 - activated1)
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


def compare_with_numerical_gradient(dW, dc, dw, db, X, one_hot_encoding_y, theta):
    """
    Compares the computed gradient with the numerical gradient using SECANT APPROXIMATION
    :param theta:   current model parameters
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
        grad_new = np.copy(grad)
        grad_new[i] += epsilon
        theta_temp = (grad_new[:4].reshape(2, 2), grad_new[4:6].reshape(1, 2), grad_new[6:8].reshape(1, 2),
                      grad_new[8].reshape(1, 1))
        L2, _, _, _, _ = compute_cost(X, one_hot_encoding_y, theta_temp)
        grad_new[i] -= epsilon
        theta_temp = (grad_new[:4].reshape(2, 2), grad_new[4:6].reshape(1, 2), grad_new[6:8].reshape(1, 2),
                      grad_new[8].reshape(1, 1))
        L1, _, _, _, _ = compute_cost(X, one_hot_encoding_y, theta_temp)
        numgrad[i] = (L2 - L1) / (2 * epsilon)
        perturb[i] = 0
        diff = np.subtract(numgrad, grad)
    return abs(diff)


def main():
    """
    Main function
    :return: None
    """
    # Load the dataset
    data = load_data()
    cols = data.shape[1]
    # set X (training data) and y (target variable)
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    # convert from data frames to numpy matrices
    X = np.array(X.values)
    y = np.array(y.values)
    one_hot_encoding_y = onehotencoding(y)
    print("X.shape = " + str(X.shape))
    print("y.shape = " + str(y.shape))
    print("one_hot_encoding_y.shape = " + str(one_hot_encoding_y.shape))

    # initialize parameters randomly
    np.random.seed(97)  # Random seed to get consistent results
    m = X.shape[0]  # number of training examples
    W = np.random.randn(2, 2)  # initialize W randomly
    c = np.random.randn(1, 2)  # initialize c randomly
    w = np.random.randn(1, 2)  # initialize w randomly
    b = np.random.randn(1, 1)  # initialize b randomly
    theta = (W, c, w, b)

    cost = []
    epochs = []

    # Run the algorithm
    for epoch in range(n_epoch):
        epochs.append(epoch)
        L, activated1, y1, activated2, y2 = compute_cost(X, one_hot_encoding_y, theta)
        cost.append(L)
        if epoch % 10000 == 1:
            print("Epoch: " + str(epoch) + " Loss: " + str(L))
        # Backward propagation
        dW, dc, dw, db = compute_grad(m, X, one_hot_encoding_y, theta, activated1, y1, activated2)
        diff = compare_with_numerical_gradient(dW, dc, dw, db, X, one_hot_encoding_y, theta)
        flag = True
        for i in range(len(diff)):
            if diff[i] > 1e-4:
                # Gradient is not correct
                flag = False
                break
            else:
                # Gradient is correct
                if epoch % 10000 == 1:
                    print("CORRECT GRADIENT", i+1)
        if not flag:  # If the computed gradient is not correct, stop the algorithm
            break
        else:
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
    print("Accuracy = ", round(100 * (1 - error)), "%")

    plt.scatter(X[:, 0], X[:, 1], c=output)
    plt.savefig("out/" + trial_name + 'scatter.png')
    plt.show()


if __name__ == '__main__':
    main()
