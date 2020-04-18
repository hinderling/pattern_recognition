# Task:
# Apply SVM to MNIST dataset
# Investigate at least two kernels e.g. linear, and RBF
# Optimize parameters using cross-validation
# Output as PDF report / ReadMe:
#  • Average accuracy during cross-validation
#  • Accuracy on the test set with optimized parameters

# Implementation:
# https://scikit-learn.org/stable/modules/svm.html


import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


def main(mode, n_train_samples, n_test_samples):
    print("Executing SVM Classifier")
    # start timer
    t0 = time.time()
    print("start importing training data...", end=" ")
    train = np.loadtxt("mnist_train.csv", delimiter=",", max_rows=n_train_samples, dtype=int)
    print("done importing training data")

    print("start importing testing data...", end=" ")
    test = np.loadtxt("mnist_test.csv", delimiter=",", max_rows=n_test_samples, dtype=int)
    print("done importing testing")

    # Using the notation proposed by scikit:
    # X = input data (i.e. Pixels), of size (n_samples, n_features)
    # y = labels/target (i.e. 0-9),  of size (n_samples)
    X_train = train[:, 1:]
    X_test = test[:, 1:]
    y_train = train[:, 0]
    y_test = test[:, 0]

    # Scale pixel values to [0, 1]
    X_test = X_test / 255
    X_train = X_train / 255

    # this block contains code for the Linear Kernel
    if mode == "linear":
        print("\nlinear Kernel selected")
        svc_lin = svm.SVC(kernel='linear')

        list_input_C = list(np.linspace(0.01, 0.05, 5))
        print("chosen C-Parameters", list_input_C)
        print(n_train_samples, "Training images were used alongside", n_test_samples, "Testing images.")

        # Initialize Grid search
        param_grid = {'C': list_input_C}  # C parameter list was chosen experimentally

        print("\nstart building grid search classifier...", end=" ")
        clf_grid = GridSearchCV(svc_lin, param_grid, cv=4, iid=True)
        print("done building grid search classifier")


        # Train the classifier on the training data (for each possible C)
        print("start fitting...", end=" ")
        clf_grid.fit(X_train, y_train)
        print("done fitting")

        dictionary_parameters = clf_grid.cv_results_["params"]
        list_input_C = [entry["C"] for entry in dictionary_parameters]
        list_output_scores = clf_grid.cv_results_["mean_test_score"]

        best_parameters = clf_grid.best_params_

        # Print results
        print("\nlist of all parameter combinations with their respective score:")
        for i in range(len(list_input_C)):
            print("C - ", list_input_C[i],
                  " = score: ", list_output_scores[i], end=" | ")
        print("\nBest Parameters:", clf_grid.best_params_)

        # proceed with the best parameter setting on the test data
        clf_best = clf_grid.best_estimator_
        print("\nselected model:", clf_best.get_params())
        print("\n### score:", clf_best.score(X_test, y_test))

        # plot
        dictionary_parameters = clf_grid.cv_results_["params"]
        list_input_C = [entry["C"] for entry in dictionary_parameters]
        list_output_scores = clf_grid.cv_results_["mean_test_score"]
        plt.plot(list_input_C, list_output_scores)
        plt.title("Linear SVM applied to the MNIST Dataset")
        plt.xlabel("C")
        plt.ylabel("Test Score")

    # This Block Contains Code for the RBF Kernel
    if mode == "rbf":
        print("\nrbf Kernel selected")
        svc_rbf = svm.SVC(kernel='rbf')

        # set parameters to test
        list_input_C = [0.1, 0.5, 1]
        list_input_gamma = [0.025, 0.05, 0.075, 0.1]
        print("chosen C-Parameters", list_input_C)
        print("chosen gamma-Parameters", list_input_gamma)
        print(n_train_samples, "Training images were used alongside", n_test_samples, "Testing images.")

        # Initialize grid search parameters
        param_grid = {'C': list_input_C, 'gamma': list_input_gamma}

        # Initialize 4-fold cross validation
        print("\nstart building grid search classifier...", end=" ")
        clf_grid = GridSearchCV(svc_rbf, param_grid, cv=4, iid=True)
        print("done building grid search classifier")

        # Train the classifier on the training data (for each parameter composition)
        print("start fitting data...", end=" ")
        clf_grid.fit(X_train, y_train)
        print("done fitting data")

        # Collect Computed Data
        dictionary_parameters = clf_grid.cv_results_["params"]
        list_output_C = [entry["C"] for entry in dictionary_parameters]
        list_output_gamma = [entry["gamma"] for entry in dictionary_parameters]
        list_output_scores = clf_grid.cv_results_["mean_test_score"]

        # Print results
        print("\nlist of all parameter combinations with their respective score:")
        for i in range(len(list_output_C)):
            print("(C - ", list_output_C[i], "/ gamma - ", list_output_gamma[i],
                  ") = score: ", list_output_scores[i], end=" | ")
        print("\nBest Parameters:", clf_grid.best_params_)

        # proceed with the best parameter setting on the test data
        clf_best = clf_grid.best_estimator_
        print("\nselected model:", clf_best.get_params())
        print("### score:", clf_best.score(X_test, y_test))

        # plot data
        counter = 0
        while counter < len(list_output_gamma):
            current_C_gammas = list_output_gamma[counter:counter + len(list_input_gamma)]
            current_C_scores = list_output_scores[counter:counter + len(list_input_gamma)]
            plt.plot(current_C_gammas, current_C_scores, label="C = " + str(list_output_C[counter]))
            plt.legend()
            counter += len(list_input_gamma)
        plt.title("RBF Kernel SVM applied to the MNIST Dataset")
        plt.xlabel("Gamma")
        plt.ylabel("Test Score")

    # print running time
    print()
    print("time: ", round(int(time.time() - t0)/60, 1), "min", sep="")

    # Show Generated plot after running everything
    if mode == "linear" or mode == "rbf":
        plt.show()


if __name__ == "__main__":
    main("rbf", 200, 20)

