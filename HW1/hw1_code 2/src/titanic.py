#!/anaconda2/bin/python


# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        """
                n,d = X.shape
        y = [self.prediction_] * n
        return y
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set

        #count how many times Survived, and how many times not Survived
        count = (y==1).sum()
        self.probabilities_ = float(count)/len(y)

        #print ("the probability of self.pro was", self.probabilities_)

        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        #>>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        n,d = X.shape
        y = []
        #y = [self.prediction_] * n
        for i in range(0, n, 1):
            p = np.random.choice(2, 1, p=[1 - self.probabilities_,
                                          self.probabilities_])  # give the weight and let them know what to pick the input from
            # do we use a loop to get an array?
            y.append(p)
        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        global num
        plt.figure(num)
        num += 1

        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    train_error = 0
    test_error = 0
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    for i in range(0,ntrials, 1):
        #get the value of the error for each division
        #train on the test data for the clf
        #test also on the data
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=i)
        #now find the error
        #first train the model
        #then predict
        #check the accuracy
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_train)
        #now find the error for the train_error
        train_err = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
        train_error += train_err

        y_pred = clf.predict(X_test)
        test_err = 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
        test_error += test_err


    #get the average
    train_error = float(train_error)/((1-test_size)*len(X))
    test_error = float(test_error)/((test_size)*len(X))
    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

num = 1
def main():
    # load Titanic dataset
    titanic = load_data("../data/titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    """
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
"""

    #========================================
    """
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)


    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')

    v = RandomClassifier()
    v.fit(X, y)
    y_pred = v.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for the Random Classifier is: %.3f' % train_error) #output is 0.485

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')

    clf = DecisionTreeClassifier(criterion="entropy") # we dont specify the other pars
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for the DecisionTreeClassifier is: %.3f' % train_error)  # output is



    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph

    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    import pydotplus

    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    graph[0].write_pdf("dtree.pdf")
    #graph.write_pdf("dtree.pdf")

    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    neighbor_size = [3,5,7]
    for size in neighbor_size:
        neigh = KNeighborsClassifier(n_neighbors=size)
        neigh.fit(X, y)
        y_pred = neigh.predict(X)
        train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
        print('\t-- training error for the KNeigbborsClassifier for n_neighbors = %d is: %.3f' % (size,train_error))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    #for each of the models, try to cross-validate

    clf = MajorityVoteClassifier()  # create MajorityVote classifier, which includes all model parameters
    train_err, test_err = error(clf, X, y, ntrials=100)
    print ("Majority: train_err: %.3f, test_err: %.3f" % (train_err, test_err))

    clf = RandomClassifier()
    train_err, test_err = error(clf, X, y, ntrials=100)
    print ("Random: train_err: %.3f, test_err: %.3f" % (train_err, test_err))

    clf = DecisionTreeClassifier(criterion="entropy")
    train_err, test_err = error(clf, X, y, ntrials=100)
    print ("DecisionTree: train_err: %.3f, test_err: %.3f" % (train_err, test_err))

    clf = KNeighborsClassifier(n_neighbors=5)
    train_err, test_err = error(clf, X, y, ntrials=100)
    print ("KNN: train_err: %.3f, test_err: %.3f" % (train_err, test_err))



    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')

    CV = 10
    #get a dictionary for each k: error
    KNN_val_arr = []
    k_arr = []
    for k in range (1,50,2):
        k_arr.append(k)
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X, y, cv=CV)
        KNN_val_arr.append(scores.mean())
    import matplotlib.pyplot as plt
    plt.plot(k_arr, KNN_val_arr, 'ro')
    plt.xlabel("K (number of neighbors) value for KNN")
    plt.ylabel("Score Value")
    plt.show()


    ### ========== TODO : END ========== ###


    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')

    """
    """
    depth_arr = range(1, 21, 1)
    train_error_arr = []
    test_error_arr = []

    for depth in range(1, 21, 1):
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        train_err, test_err = error(clf, X, y, ntrials=100)
        train_error_arr.append(train_err)
        test_error_arr.append(test_err)

    plt.plot(depth_arr, train_error_arr, 'b', label="train_error")
    plt.plot(depth_arr, test_error_arr, 'r', label="test_error")
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=-5.0)
    plt.legend(bbox_to_anchor=(1, 0), loc=2, borderaxespad=-5.0)

    plt.xlabel("Tree depth")
    plt.ylabel("Error")
    plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')

    """

    """
    Another useful tool for evaluating classifiers is learning curves, which show how classifier performance (e.g. error)
     relates to experience (e.g. amount of training data). For this experiment, first generate a random 90/10 split of the training data and
      do the following experiments considering the 90% fraction as training and 10% for testing.
    Run experiments for the decision tree and k-nearest neighbors classifier with the best depth limit and k value you found above.
    This time, vary the amount of training data by starting with splits of 0.10 (10% of the data from 90% fraction) and 
    working up to full size 1.00 (100% of the data from 90% fraction) in increments of 0.10. 
    Then plot the decision tree and k-nearest neighbors training and test error against the amount of training data. 
    Include this plot in your writeup, and provide a 1-2 sentence description of your observations.
    """
    #first use train_and_test
    #90% training, 10% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    best_depth = 3
    best_k = 5

    tree_train_err_arr = []
    tree_test_err_arr = []
    knn_train_err_arr = []
    knn_test_err_arr = []
    increment_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in increment_arr:
        train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=1-i)
        #make the model for both decisionTree
        clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth)
        train_err, test_err = error(clf_tree, train_x, train_y, ntrials=100)
        tree_train_err_arr.append(train_err)
        tree_test_err_arr.append(test_err)

        clf_knn = KNeighborsClassifier(n_neighbors=best_k)
        train_err, test_err = error(clf_knn, train_x, train_y, ntrials=100)
        knn_train_err_arr.append(train_err)
        knn_test_err_arr.append(test_err)

    plt.plot(increment_arr, tree_train_err_arr, 'b', label="tree train error")
    plt.plot(increment_arr, tree_test_err_arr, 'g', label="tree test error")
    plt.plot(increment_arr, knn_train_err_arr, 'y', label="KNN train error")
    plt.plot(increment_arr, knn_test_err_arr, 'r', label="KNN test error")
    # Place a legend to the right of this smaller subplot.
    plt.legend()
    plt.xlabel("Fraction of training sample")
    plt.ylabel("Error")
    plt.show()



    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
