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
degree = 3  # p, order of model
beta = 0.01  # regularization coefficient
trial_name = 'p' + str(degree) + '_fit_beta_' + str(beta)  # will add a unique sub-string to output of this program
alpha = 0.01  # step size coefficient
eps = 0.0  # controls convergence criterion
n_epoch = 1000  # number of epochs (full passes through the dataset)


# begin simulation

def regress(X, theta):
    y = []
    for i in range(len(X)):
        res = float(theta[0])
        for j in range(len(X[i])):
            res += X[i][j] * float(theta[1][0][j])
        y.append(res)
    return np.array(y)


############################################################################

def gaussian_log_likelihood(mu, y, theta):
    return (calculate_f_theta(mu, theta) - float(y)) ** 2


############################################################################

def computeCost(X, y, theta, beta):  ## loss is now Bernoulli cross-entropy/log likelihood
    m = len(X)
    loss = 0.0
    regularization = 0.0
    for i in range(m):
        loss += gaussian_log_likelihood(X[i], y[i], theta)
    for j in range(len(theta[1])):
        regularization += theta[1][j]**2
    regularization += theta[0]
    return (loss / (2 * m)) + (beta * regularization / (2 * m))


def calculate_f_theta(x, theta):
    vector_length = len(x)
    result = float(theta[0])
    for i in range(vector_length):
        a = float(x[i])
        b = float(theta[1][0][i])
        result += float(x[i]) * float(theta[1][0][i])
    return result


def computeGrad(X, y, theta, beta):
    dL_dfy = 0.0  # derivative w.r.t. to model output units (fy)
    dL_db = 0.0  # derivative w.r.t. model weights w
    dL_dw = 0.0  # derivative w.r.t model bias b

    m = len(X)
    dL_dw_list = list()

    for w in theta[1][0]:
        dL_dw = 0.0
        for i in range(m):
            f_theta = calculate_f_theta(X[i], theta)
            dL_db += (f_theta - float(y[i]))
            for j in range(len(X[i])):
                dL_dw += (f_theta - float(y[i])) * float(X[i][j]) + (beta/m) * theta[1][0][j]
        dL_dw_list.append(dL_dw / m)

    nabla = (dL_db/m, dL_dw_list)  # nabla represents the full gradient
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

# convert from data frames to numpy matrices
X = np.array(X.values)
length = len(X)
X_list = []
for item in X:
    datapoint = []
    for i in range(degree):
        datapoint.append(float(item) ** (i + 1))
    X_list.append(datapoint)
X = np.array(X_list)  # Converted to a polynomial function

y = np.array(y.values)

# convert to numpy arrays and initialize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta, beta)
halt = 0  # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
cost = []
epochs = []
while i < n_epoch and halt == 0:
    epochs.append(i)
    dL_db, dL_dw = computeGrad(X, y, theta, beta)
    theta_zero = float(theta[0]) - alpha * dL_db
    theta_one_to_p = []
    for j in range(len(theta[1][0])):
        theta_one_to_p.append(float(theta[1][0][j]) - alpha * dL_dw[j])
    theta = (np.array([theta_zero]), np.array([theta_one_to_p]))
    b = theta[0]
    w = theta[1]
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
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test,
                        axis=1)  # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
X_list = []
for item in X_feat:
    datapoint = []
    for i in range(degree):
        datapoint.append(float(item) ** (i + 1))
    X_list.append(datapoint)
X_feat = np.array(X_list)  # Converted to a polynomial function

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################

plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
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

############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################

plt.show()
