#   Task:
# • Apply MLP to the MNIST dataset
# • MLP with one hidden layer
# • Optimize parameters using cross-validation
# • Discuss a good architecture for your framework so that you can reuse software components in later exercises.
# • Output as PDF report / ReadMe
#   • Plot showing error-rate on training and validation set
#     with respect to the training epochs
#   • Accuracy on the test set with optimized parameters
#   • Average accuracy during cross-validation for all investigated kernels (e.g. linear and RBF) and all parameter values (e.g. C and γ ).


# Based on: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Load the datasset
print("Loading training data ...")
trainData = np.loadtxt(open("dataset/mnist_train.csv","r"), delimiter=",", skiprows=0, max_rows=10000)
print("Loading test data ...")
testData = np.loadtxt(open("dataset/mnist_test.csv","r"), delimiter=",", skiprows=0, max_rows=10000)

#Using the notation proposed by scikt:
# X = input data (i.e. Pixels), of size (n_samples, n_features)
# y = labels/target (i.e. 0-9),  of size (n_samples)
y_train  = trainData[:,0]  
y_test   = testData [:,0]
X_train  = trainData[:,1:]
X_test   = testData [:,1:]

# Multi-layer Perceptron is sensitive to feature scaling, so it is highly 
# recommended to scale your data. E.g, scale each attribute 
# on the input vector X to [0, 1]
X_train = X_train / 255
X_test  = X_test  / 255

#Hyperparameters to validate:
# - (Hidden layers = 1)
# - nb. neurons/hidden layer = [10,100]
# - learning rate = [0.001, 0.1]
# - nb. training iterations 
# - random initialization, choose best one for validation
# for grid search there is a scikit library:
# https://scikit-learn.org/stable/modules/grid_search.html

#With fixed parameters
run = 0
if (run):
    #initialize the classifier
    mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=100, alpha=1e-4,
                   solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

    #train the model on the training data and print score
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))

    #fit the test data and print the score
    print("Test set score: %f" % mlp.score(X_test, y_test))


#With multiple parameters: gridsearch
# Specify:
    #1: Estimator (i.e. MLP)
    #2: Parameter space (e.g nb neurons, learning rates, ...)
    #3: method for searching or sampling candidates;
    #4: a cross-validation scheme
    #5: a score function.


# 1: Specify the estimator without parameters 
mlp = MLPClassifier()

# 2: Specify the parameters. Use mlp.get_params() to get estimator parameters
# specifiable in GridSearchCV
# GridSearcCV works exhaustively (tests all combinations possible)
# We escape unwanted combinations by defining mulitple grids, 
# using multiple "{}" containers
param_grid = [
  {'hidden_layer_sizes': np.linspace(10,100,num=10),'max_iter'=[100], 'alpha'=[1e-4],
                    solver=['sgd'], 'verbose'=[10], 'random_state'=[1],
                    'learning_rate_init'=np.logspace(0.001,0.1)}
 ]


# 3: init the gridsearch with the estimator, and pass it the training data
# Multiple threads: define n_jobs
search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=6)
search.fit(X, y)



#function to print weight functions:
#https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py


fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()