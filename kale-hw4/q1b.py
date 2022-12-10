import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# meta-parameters for program
prob = "Q1B_"
beta = 0.000001  # regularization coefficient or lambda
alpha = 0.03  # step size coefficient or learning rate
eps = 0.000001  # controls convergence criterion
n_epoch = 124900  # number of epochs (full passes through the dataset)
trial_name = prob + "beta:" + str(beta) + "alpha:" + str(
    alpha) + "n_epochs:" + str(n_epoch)  # will add a unique sub-string to output of this program
epsilon = 0.001  # secant approximation


def load_data():
    """
    Loads the dataset
    :return: The dataset
    """
    path = os.getcwd() + '/spiral_train.dat'
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


# def onehotencoding(y):
#     """
#     One-hot encodes the labels
#     :param y: Labels
#     :return: One-hot encoded labels
#     """
#     y = y.reshape(-1)
#     n_values = np.max(y) + 1
#     return np.eye(n_values)[y]


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
    Computes the softmax function for the output layer
    :param x: Input
    :return: Softmax function output
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def forward_propagation(X, theta):
    """
    Computes the forward propagation
    :param X: Dataset
    :param theta: Model parameters
    :return: Forward propagation output
    """
    W, c, w, b = theta
    y1 = np.dot(X, W.T) + c
    activated1 = sigmoid(y1)
    y2 = np.dot(activated1, w.T) + b
    activated2 = sigmoid(y2)
    return activated1, y1, activated2, y2


def compute_cost(X, y, theta):
    """
    Computes the cost function value
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
        theta_temp = (theta[0] - perturb[:6].reshape(3, 2), theta[1] - perturb[6:9].reshape(1, 3),
                      theta[2] - perturb[9:18].reshape(3, 3), theta[3] - perturb[18:].reshape(1, 3))
        L1, _, _, _, _ = compute_cost(X, one_hot_encoding_y, theta_temp)
        theta_temp = (theta[0] + perturb[:6].reshape(3, 2), theta[1] + perturb[6:9].reshape(1, 3),
                      theta[2] + perturb[9:18].reshape(3, 3), theta[3] + perturb[18:].reshape(1, 3))
        L2, _, _, _, _ = compute_cost(X, one_hot_encoding_y, theta_temp)
        numgrad[i] = (L2 - L1) / (2 * epsilon)
        perturb[i] = 0
    return np.amin(np.abs(np.subtract(numgrad, grad)))


def plot_decision_boundary(theta, X, y):
    """
    Plots the decision boundary
    :param param: Model parameters
    :param X: Dataset
    :param y: Labels
    """
    # Set min and max values and give it some padding
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    resolution = 0.01
    # Generate a grid of points with distance resolution between them
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # Predict the function value for the whole gid
    Z = predict(np.c_[xx1.ravel(), xx2.ravel()], theta)
    Z = Z.reshape(xx1.shape)
    # Plot the contour and training examples
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.RdYlGn)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlGn)
    plt.show()


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
    one_hot_encoding_y = one_hot_encoding(y)
    print("X.shape = " + str(X.shape))
    print("y.shape = " + str(y.shape))
    print("one_hot_encoding_y.shape = " + str(one_hot_encoding_y.shape))

    # np.random.seed(654)
    np.random.seed(4)  # Random seed to get consistent results
    m = X.shape[0]  # number of training examples
    W = np.random.randn(3, 2)  # initialize W randomly
    # c = np.random.randn(1, 3)  # initialize c randomly
    c = np.ones((1, 3))  # initialize c randomly
    w = np.random.randn(1, 3)  # initialize w randomly
    # b = np.random.randn(1, 1)  # initialize b randomly
    b = np.zeros((1, 1))  # initialize b randomly
    theta = (W, c, w, b)

    cost = []
    epochs = []

    np.set_printoptions(suppress=True)

    # Run the algorithm
    for epoch in range(n_epoch):
        epochs.append(epoch)
        L, activated1, y1, activated2, y2 = compute_cost(X, one_hot_encoding_y, theta)
        cost.append(L)
        if epoch % 1000 == 1:
            print("Epoch: " + str(epoch) + " Loss: " + str(L))
        # Backward propagation
        dW, dc, dw, db = compute_grad(m, X, one_hot_encoding_y, theta, activated1, y1, activated2)
        # diff = check_grad(dW, dc, dw, db, X, one_hot_encoding_y, theta)
        diff = 0.0000001
        if diff > 1e-4:
            print("Gradient check failed")
            break
        else:
            # if epoch % 10000 == 1:
            #     print("Gradient check passed")
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

    # plt.scatter(X[:, 0], X[:, 1], c=output)
    # plt.savefig("out/" + trial_name + 'scatter.png')
    plot_decision_boundary(theta, X, y)


if __name__ == '__main__':
    main()
