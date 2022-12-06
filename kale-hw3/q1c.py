import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
CSCI 635: Introduction to Machine Learning
HW3 Q1C

@author/lecturer - Kushal S. Kale

"""

# meta-parameters for program
prob = "Q1C_"
degree = 11  # p, order of model
beta = 1  # regularization coefficient
alpha = 0.0001  # step size coefficient
eps = 0.000001  # controls convergence criterion
n_epoch = 400000  # number of epochs (full passes through the dataset)
trial_name = prob + "beta:" + str(beta) + "alpha:" + str(
    alpha) + "n_epochs:" + str(n_epoch)  # will add a unique sub-string to output of this program
epsilon = 0.001  # secant approximation


def regress(X, theta):
    """
    Computes the model output
    :param X: Dataset
    :param theta: Model parameters
    :return: Model output
    """
    b = theta[0]
    w = theta[1]
    result = np.add(b, np.dot(X, w))
    return result


def bernoulli_log_likelihood(p, y):
    """
    Computes the Bernoulli log likelihood
    :param p: Posterior probability
    :param y: Labels
    :return: Bernoulli log likelihood
    """
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def posterior(f_theta):
    """
    Computes the posterior probability of the label being 1
    :param f_theta: Model output
    :return: Posterior probability
    """
    return np.exp(f_theta) / np.sum(np.exp(f_theta))


def compute_cost(X, y, theta, beta):
    """
    Computes the Bernoulli cross-entropy/log likelihood loss function
    :param X: Dataset
    :param y: Labels
    :param theta: Model parameters
    :param beta: Regularization coefficient
    :return: Loss
    """
    m = len(X)
    f_theta = regress(X, theta)
    # loss = np.sum(bernoulli_log_likelihood(posterior(f_theta), y))
    # regularization = np.sum(np.square(theta[1]))
    # print("y_shape", y.shape)
    # print(y)
    # print("l_shape", np.log(posterior(f_theta)).shape)
    loss = np.sum(np.sum(np.multiply(y, np.log(posterior(f_theta)))))
    regularization = np.sum(np.square(theta[1]))
    return (-1 * loss / m) + (beta * regularization / 2)


def compute_grad(X, y, theta, beta):
    """
    Computes the gradient of the loss function with respect to the model parameters
    :param X: Dataset
    :param y: Labels
    :param theta: Model parameters
    :param beta: Regularization coefficient
    :return: Gradient of the loss function
    """
    f_theta = regress(X, theta)
    dL_dfy = posterior(f_theta) - y  # derivative w.r.t. to model output units (fy)
    dL_db = 0.0  # derivative w.r.t. model bias
    dL_dw = list()  # derivative w.r.t model weights w
    # dL_dw_list = list()  # derivative w.r.t model weights w

    m = len(X)

    # dl_dw = np.dot(dL_dfy.T, X) + beta * theta[1]
    # print("dot", np.dot(X.T, dL_dfy / m))
    # print("dot shape", np.dot(X.T, dL_dfy / m).shape)
    # print("beta", (beta * theta[1]))
    # print("beta shape", (beta * theta[1]).shape)
    dL_dw = np.dot(X.T, dL_dfy / m) + (beta * theta[1])
    # dL_dw = (1 / m) * np.dot(dL_dfy.T, X) + beta * theta[1]
    # dL_dw = np.product(X.T, np.divide(dL_dfy, m)) + beta * theta[1]
    dL_db = np.sum(dL_dfy) / m

    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    return nabla


def predict(X, theta):
    """
    Predicts the class labels
    :param X: Dataset
    :param theta: Model parameters
    :return: Predicted class labels
    """
    f_theta = regress(X, theta)
    # print("THETA: ", f_theta)
    # print("TEST: ", np.argmax(f_theta, axis=1))
    return np.argmax(f_theta, axis=1)


def check_gradient(X, y, theta, beta):
    """
    Checks the gradient computed by the compute_grad function
    :param X: Dataset
    :param y: Labels
    :param theta: Model parameters
    :param beta: Regularization coefficient
    :return: The secant approximation of the derivatives
    """
    scalar_derivatives = []
    vec = theta[1].flatten()
    vec = np.append(vec, theta[0])
    for i in range(len(vec)):
        val = vec[i]
        vec[i] = val + epsilon
        cost1 = compute_cost(X, y, (vec[0], vec[1:].reshape(theta[1].shape)), beta)
        vec[i] = val - epsilon
        cost2 = compute_cost(X, y, (vec[0], vec[1:].reshape(theta[1].shape)), beta)
        scalar_derivatives.append((cost1 - cost2) / (2 * epsilon))
        vec[i] = val
    return scalar_derivatives


def one_hot_encoding(y):
    """
    One-hot encodes the labels
    :param y: Labels
    :return: One-hot encoded labels
    """
    y = y.reshape(-1)
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


def main():
    """
    Main function
    :return: None
    """
    # begin simulation
    path = os.getcwd() + '/iris_train.dat'
    data = pd.read_csv(path, header=None)

    ############################################################################
    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    # print("X:", X)
    # print("Y:", y)

    # convert from data frames to numpy matrices
    X = np.array(X.values)
    y = np.array(y.values)
    onehotencoding_y = one_hot_encoding(y)

    # convert to numpy arrays and initialize the parameter array theta
    # w = np.ones((X.shape[1], 3))
    w = np.array(([7, 2, 8], [8, 2, 9], [5, 1, 6], [5, 2, 3]))
    # print("W:", w)
    b = np.array([0.5, 0.5, 0.5])
    # print("B:", b)
    theta = (b, w)

    L_list = []

    L = compute_cost(X, onehotencoding_y, theta, beta)
    print("-1 L = {0}".format(L))
    L_best = L
    tolerance = 5  # halting variable (you can use these to terminate the loop if you have converged)
    i = 0
    halt = 0
    cost = []  # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
    epochs = []
    while i < n_epoch and halt == 0:
        epochs.append(i)
        dL_db, dL_dw = compute_grad(X, onehotencoding_y, theta, beta)
        theta = (theta[0] - alpha * dL_db, theta[1] - alpha * dL_dw)
        b = theta[0]
        w = theta[1]
        L = compute_cost(X, onehotencoding_y, theta, beta)  # track our loss after performing a single step
        cost.append(L)

        if L < L_best:
            L_best = L

        # if L_best - L < eps:
        #     tolerance -= 1
        #
        # if tolerance == 0:
        #     halt = 1
        # check_gradient(X, y, theta, beta)
        i += 1

    # print parameter values found after the search
    print("w = ", w)
    print("b = ", b)

    error = 0
    output = predict(X, (b, w))
    # print("output", output)
    for i in range(len(output)):
        if output[i] == y[i]:
            error += 1
    error /= len(y)
    print("Training Accuracy = ", round(100 * (1 - error)), "%")
    plt.scatter(X[:, 0], X[:, 1], c=output)
    plt.savefig("out/" + trial_name + 'scatter.png')
    plt.show()

    theta = (b, w)

    path = os.getcwd() + '/iris_test.dat'
    data = pd.read_csv(path, header=None)

    ############################################################################
    # set X (training data) and y (target variable)
    cols = data.shape[1]
    Xt = data.iloc[:, 0:cols - 1]
    yt = data.iloc[:, cols - 1:cols]

    # convert from data frames to numpy matrices
    Xt = np.array(Xt.values)
    yt = np.array(yt.values)
    onehotencoding_yt = one_hot_encoding(yt)

    output = predict(Xt, theta)
    # print("Output:", output)
    # print(list([int(i) for i in yt]))

    # print("Predicted class probabilities = ", output)
    error = 0
    for i in range(len(output)):
        if output[i] == yt[i]:
            error += 1
    error /= len(yt)
    print("Validation Accuracy = ", round(100 * (1 - error)), "%")

    plt.plot(epochs, cost)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.savefig("out/" + trial_name + '_cost_vs_epoch.png')
    plt.show()

    plt.scatter(Xt[:, 0], Xt[:, 1], c=output)
    plt.savefig("out/" + trial_name + 'scatter.png')
    plt.show()


main()
