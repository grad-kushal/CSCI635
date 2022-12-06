import math
import os

import np as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 3: Multivariate Regression & Classification

@author/lecturer - Kushal S. Kale

'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p6_reg0'  # will add a unique sub-string to output of this program
degree = 6  # p, degree of model (PLEASE LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 1.0  # regularization coefficient
alpha = 0.01  # step size coefficient
n_epoch = 2000  # number of epochs (full passes through the dataset)
eps = 0.0001  # controls convergence criterion


# begin simulation

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict(X, theta):
    pred_list = []
    m = len(X)
    for i in range(m):
        pred_list.append(1 if float(regress(X[i], theta)) > 0.5 else 0)
    return np.array(pred_list)


def regress(X, theta):
    b = theta[0]
    w = theta[1]
    return b + np.dot(X, w.T)


def bernoulli_log_likelihood(X, y):
    fthetaa = calculate_f_theta(X)
    return -1 * (y * np.log(fthetaa) + (1 - y) * np.log(1 - fthetaa))


def calculate_f_theta(x):
    vector_length = len(x)
    result = float(theta2[0])
    for i in range(vector_length):
        a = float(x[i])
        b = float(theta2[1][0][i])
        result += float(x[i]) * float(theta2[1][0][i])
    return sigmoid(result)


def computeCost(X, y, theta, beta):  ## loss is now Bernoulli cross-entropy/log likelihood
    m = len(X)
    loss = 0.0
    regularization = 0.0
    for i in range(m):
        loss += bernoulli_log_likelihood(X[i], y[i])
    for j in range(len(theta[1])):
        regularization += float(theta[1][0][j]) ** 2
    regularization += float(theta[0])
    return (loss / m) + (beta * regularization / (2 * m))


def computeGrad(X, y, theta, beta):
    dL_dfy = None  # derivative w.r.t. to model output units (fy)
    dL_db = 0.0  # derivative w.r.t. model weights w
    dL_dw = None  # derivative w.r.t model bias b

    m = len(X)
    n = len(theta[1][0])
    dL_dw_list = list()

    for j in range(n):
        dL_dw = 0.0
        for i in range(m):
            f_theta = calculate_f_theta(X[i])
            dL_db += (f_theta - float(y[i]))
            for k in range(len(X[i])):
                dL_dw += (f_theta - float(y[i])) * float(X[i][j]) + (beta / m) * theta[1][0][j]
        dL_dw_list.append(dL_dw / m)
    nabla = (dL_db / m, dL_dw_list)  # nabla represents the full gradient
    return nabla


############################################################################

path = os.getcwd() + '/data/prob3.dat'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
stats = data2.describe()
print("STATS ABOUT THE DATA: \n" + str(stats))

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

x1 = data2['Test 1']
x2 = data2['Test 2']

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree + 1):
    for j in range(0, i + 1):
        data2['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
        cnt += 1

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]

# convert to numpy arrays and initialize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
w = np.zeros((1, X2.shape[1]))
b = np.array([0])
theta2 = (b, w)

L = computeCost(X2, y2, theta2, beta)
L_minimum = L
halt = 4  # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
cost = []
epochs = []
while i < n_epoch and halt != 0:
    epochs.append(i)
    dL_db, dL_dw = computeGrad(X2, y2, theta2, beta)
    theta_zero = float(theta2[0]) - alpha * dL_db
    theta_one_to_p = []
    for j in range(len(theta2[1][0])):
        theta_one_to_p.append(float(theta2[1][0][j]) - alpha * dL_dw[j])
    theta2 = (np.array([theta_zero]), np.array([theta_one_to_p]))
    b = theta2[0]
    w = theta2[1]
    L = computeCost(X2, y2, theta2, beta)
    if eps > abs(float(L_minimum) - float(L)):
        halt -= 1
        print("here halt: " + str(halt))
        print(i)
    else:
        halt = 4
    if float(L) < float(L_minimum):
        L_minimum = float(L)
    cost.append(float(L))

    i += 1

# print parameter values found after the search
print("w = ", w)
print("b = ", b)

predictions = predict(X2, theta2)
# compute error (100 - accuracy)
err = 0.0
correct = 0.0
for index in range(len(predictions)):
    if float(predictions[index]) == float(y2[index]):
        correct += 1
err = (len(predictions) - correct) / len(predictions)

print('Error = {0}%'.format(err * 100.))

# make contour plot input data
xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to input x1 & x2
for i in range(1, degree + 1):
    for j in range(0, i + 1):
        feat = np.power(xx1, i - j) * np.power(yy1, j)
        if len(grid_nl) > 0:
            grid_nl = np.c_[grid_nl, feat]
        else:
            grid_nl = feat
probs = regress(grid_nl, theta2).reshape(xx.shape)

# create contour plot to visualize decision boundaries of model above
f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
ax.scatter(x1, x2, s=50, c=np.squeeze(y2),
           cmap="RdBu",
           vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
## plot done...ready for using/saving

############################################################################
# WRITEME: write your code here to model to save this plot to disk 
#          (look up documentation or the inter-webs for matplotlib)
############################################################################
plt.savefig("out/prob_3_degree_" + str(degree) + "contour_plot_beta_" + str(beta) + ".png")
plt.show()

plt.plot(epochs, cost)
plt.savefig("out/prob_3_degree_" + str(degree) + "cost_vs_epochs" + str(beta) + ".png")
plt.show()
