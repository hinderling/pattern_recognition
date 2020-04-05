#   Task:
# done • Apply MLP to the MNIST dataset
# done • MLP with one hidden layer
# done • Optimize parameters using cross-validation

# todo • Discuss a good architecture for your framework so that you can reuse software components in later exercises.
# todo • Output as PDF report / ReadMe
#    todo • Plot showing error-rate on training and validation set
#     with respect to the training epochs
#   todo • Accuracy on the test set with optimized parameters
#   todo • Average accuracy during cross-validation for all investigated kernels (e.g. linear and RBF) and all parameter values (e.g. C and γ ).


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

#Using the notation proposed by scikit:
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

#cast label info to int
y_test = y_test.astype(int)
y_train = y_train.astype(int)



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

#Hyperparameters to validate:
# - (Hidden layers = 1)
# - nb. neurons/hidden layer = [10,100]
# - learning rate = [0.001, 0.1]
# - nb. training iterations 
# - random initialization, choose best one for validation
# for grid search there is a scikit library: GridSearchCV
# https://scikit-learn.org/stable/modules/grid_search.html

# GridSearchCV:
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
# We escape unwanted combinations by defining mulitple search grids, 
# using one "{}" container for each grid
param_grid = [
    {'hidden_layer_sizes': np.rint(np.logspace(0.8,2.1,10)).astype('int'),
    'max_iter':[50], 'alpha':[1e-4],  'solver':['sgd'], 'verbose':[10], 'random_state':[1],
                    'learning_rate_init':np.logspace(-4,-0.5,10)}
 ]


# 3, 4: init the gridsearch with the estimator, and pass it the training data
# - Multiple threads: define n_jobs
# - Cross validation: define cv, hoW many groups should be made? (default is 5)
#     see: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
search = GridSearchCV(mlp, param_grid, cv=4)

search.fit(X_test, y_test)


# 5: Score function: Run all models the test dataset
search.score(X_test, y_test)
#to get just the best configuration, use: 
mlp = search.best_estimator_
mlp.get_params()
#if we are intrested in how the models compare, we can find all the test results in a dict
search.cv_results_
#lets say we want to plot the score as a function of nb. neurons and learning rate:

hidden_layers = search.cv_results_['param_hidden_layer_sizes'].data #masked array
learning_rate = search.cv_results_['param_learning_rate_init'].data #masked array
test_score = search.cv_results_['mean_test_score'] #array
print(hidden_layers)
print(learning_rate)
print(test_score)




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