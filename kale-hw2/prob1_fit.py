import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 1: Univariate Regression

@author/lecturer - Kushal S. Kale

'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.01  # step size coefficient
eps = 0.00000  # controls convergence criterion
n_epoch = 1800  # number of epochs (full passes through the dataset)


# begin simulation

def regress(X, theta):
    print(type(X[0][0]))
    y = []

    for i in range(len(X)):
        y.append(X[i][0]*float(theta[1]) + float(theta[0]))
    return np.array(y)


############################################################################

def gaussian_log_likelihood(X, y, theta):
    return (calculate_f_theta(float(X), theta) - float(y)) ** 2


def calculate_f_theta(x, theta):
    return (float(theta[0]) + x*float(theta[1]))


def computeCost(X, y, theta):  # loss is now Bernoulli cross-entropy/log likelihood
    m = len(X)
    result = 0.0
    for i in range(m):
        result += gaussian_log_likelihood(X[i], y[i], theta)

    return result/(2*m)


############################################################################

def computeGrad(X, y, theta):
    dL_dfy = 0.0  # derivative w.r.t. to model output units (fy)
    dL_db = 0.0  # derivative w.r.t. model bias b
    dL_dw = 0.0  # derivative w.r.t model weights w
    m = len(X)
    for i in range(m):
        dL_dw += (calculate_f_theta(float(X[i]), theta) - float(y[i])) * float(X[i])
        dL_db += (calculate_f_theta(float(X[i]), theta) - float(y[i]))
    nabla = (dL_db/m, dL_dw/m)  # nabla represents the full gradient
    return nabla


############################################################################

path = os.getcwd() + '/data/prob1.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

stats = data.describe()
print("STATS ABOUT THE DATA: \n" + str(stats))
plotX = []
plotY = []
for i in range(len(data)):
    plotX.append(data.loc[i].at["X"])
    plotY.append(data.loc[i].at["Y"])
scatter_plot = plt.scatter(plotX, plotY)
plt.savefig("out/prob1_data_scatter_plot.png")
plt.show()


############################################################################

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

# convert to numpy arrays and initialize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = (b, w)
L_list = []

L = computeCost(X, y, theta)

print("-1 L = {0}".format(L))
L_best = L
halt = 0  #halting variable (you can use these to terminate the loop if you have converged)
i = 0
cost = []  # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
epochs = []
while (i < n_epoch and halt == 0):
    epochs.append(i)
    dL_db, dL_dw = computeGrad(X, y, theta)
    theta = (float(theta[0]) - alpha * dL_db, float(theta[1]) - alpha * dL_dw)
    b = theta[0]
    w = theta[1]
    L = computeCost(X, y, theta)  # track our loss after performing a single step
    cost.append(L)
    # print(" {0} L = {1}".format(i, L))
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)
print("Printing cost list: ", cost)

kludge = 0.25  # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")

############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
plt.savefig("out/prob1_regression_test_data.png")
plt.show()
############################################################################


plt.plot(cost, epochs)
plt.savefig("out/prob1_cost_vs_epochs")
plt.show()  # convenience command to force plots to pop up on desktop
