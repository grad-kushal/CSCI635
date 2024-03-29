import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 2: Polynomial Regression &

@author/lecturer - Kushal S. Kale

'''

# NOTE: you will need to tinker with the meta-parameters below yourself
#       (do not think of them as defaults by any means)
# meta-parameters for program
prob = "prob2_"
degree = 11 # p, order of model
beta = 0.0  # regularization coefficient
trial_name = 'p' + str(degree) + '_fit_beta_' + str(beta)  # will add a unique sub-string to output of this program
alpha = 0.01  # step size coefficient
eps = 0.0  # controls convergence criterion
n_epoch = 1000  # number of epochs (full passes through the dataset)


# begin simulation

def regress(X, theta):
    b = theta[0]
    w = theta[1]
    result = np.add(b, np.dot(X, w.T))
    return result


############################################################################

def gaussian_log_likelihood(mu, y, theta):
    regress_val = regress(mu, theta)
    result = np.square(regress_val - y)
    return result


############################################################################

def computeCost(X, y, theta, beta):  # loss is now Bernoulli cross-entropy/log likelihood
    m = len(X)
    # loss = np.sum(gaussian_log_likelihood(X, y, theta))
    loss = np.sum(np.square(regress(X, theta) - y))
    regularization = np.sum(np.square(theta[1]))

    # for i in range(m):
    #     loss += float(gaussian_log_likelihood(X[i], y[i], theta))
    # for j in range(len(theta[1])):
    #     regularization += float(theta[1][0][j])**2
    # regularization += float(theta[0])
    return (loss / (2 * m)) + (beta * regularization / (2 * m))


# def calculate_f_theta(x, theta):
#     vector_length = len(x)
#     result = float(theta[0])
#     for i in range(vector_length):
#         a = float(x[i])
#         b = float(theta[1][0][i])
#         result += float(x[i]) * float(theta[1][0][i])
#     return result


def computeGrad(X, y, theta, beta):
    dL_dfy = 0.0  # derivative w.r.t. to model output units (fy)
    dL_db = None # derivative w.r.t. model weights w
    dL_dw = 0.0  # derivative w.r.t model bias b

    m = len(X)
    b = theta[0]
    w = theta[1]

    dL_dw = np.sum(np.multiply(regress(X, theta) - y, X)) + np.multiply(beta/m, w)
    dL_db = np.sum(np.multiply(regress(X, theta) - y, 1))
    # dL_dw_list = list()
    #
    # for w in theta[1][0]:
    #     dL_dw = 0.0
    #     for i in range(m):
    #         f_theta = regress(X[i], theta)
    #         dL_db += (f_theta - float(y[i]))
    #         for j in range(len(X[i])):
    #             dL_dw += (f_theta - float(y[i])) * float(X[i][j]) + (beta/m) * theta[1][0][j]
    #     dL_dw_list.append(dL_dw / m)

    nabla = (dL_db/m, dL_dw/m)  # nabla represents the full gradient
    return nabla


############################################################################

path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

stats = data.describe()
print("STATS ABOUT THE DATA: \n" + str(stats))
plotX = []
plotY = []
for i in range(len(data)):
    plotX.append(data.loc[i].at["X"])
    plotY.append(data.loc[i].at["Y"])
scatter_plot = plt.scatter(plotX, plotY)
plt.savefig("out/" + prob + trial_name + '_data_scatter_plot.png')
plt.show()


# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

cnt = 0
for i in range(1, degree + 1):
        data['X' + str(i)] = np.power(X, i)
        cnt += 1

# set X and y
data.insert(0, 'Y', data.pop('Y'))
data.pop('X')
cols = data.shape[1]
X2 = data.iloc[:, 1:cols]
y2 = data.iloc[:, 0:1]


# convert from data frames to numpy matrices
X = np.array(X2.values)
y = np.array(y2.values)

# initialize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = (b, w)

X_temp = np.copy(X)

L = computeCost(X, y, theta, beta)
halt = 0  # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
cost = []
epochs = []
while i < n_epoch and halt == 0:
    epochs.append(i)
    dL_db, dL_dw = computeGrad(X, y, theta, beta)
    b = theta[0]
    w = theta[1]
    b = np.subtract(b, np.multiply(alpha, dL_db))
    w = np.subtract(w, np.multiply(alpha, dL_dw))
    theta = (b, w)
    # theta_zero = float(theta[0]) - alpha * float(dL_db)
    # theta_one_to_p = []
    # for j in range(len(theta[1][0])):
    #     theta_one_to_p.append(float(theta[1][0][j]) - alpha * float(dL_dw[j]))
    # theta = (np.array(theta_zero), np.array([theta_one_to_p]))
    # b = float(theta[0])
    # w = theta[1]
    L = computeCost(X, y, theta, beta)
    cost.append(L)

    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################

    # print(" {0} L = {1}".format(i, L))
    # print(i)
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X1.min(), data.X1.max(), 100)
X_feat = np.expand_dims(X_test,
                        axis=1)  # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
X_list = []
for item_number in range(X_feat.shape[0]):
    datapoint = []
    for order in range(1, degree+1):
        # datapoint.append(X_feat[item_number] ** order)
        datapoint.append(math.pow(X_feat[item_number], order))
    X_list.append(datapoint)
X_feat = np.array(X_list)  # Converted to a polynomial function


############################################################################

plt.scatter(X_temp[:, 0], y, edgecolor='r', s=20, label="Samples")
plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
plt.savefig("out/" + prob + trial_name + '_regression_test_data.png')
plt.show()


plt.plot(cost, epochs)
plt.savefig("out/" + prob + trial_name + '_cost_vs_epoch.png')
plt.show()

