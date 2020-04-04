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
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np

# Load data from https://www.openml.org/d/554
#X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#X = X / 255.

# Load the dataset
print("Loading training data ...")
trainData = np.loadtxt(open("dataset/mnist_train.csv","r"), delimiter=",", skiprows=0, max_rows=3000)
print("Loading test data ...")
testData = np.loadtxt(open("dataset/mnist_test.csv","r"), delimiter=",", skiprows=0, max_rows=3000)

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

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))




fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()