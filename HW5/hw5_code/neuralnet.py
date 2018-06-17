'''The code is modified from Stanford CS231
'''

# A bit of setup
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(0) # do NOT change this line

def load_data(N,D,K):
    X = np.zeros((N*K,D))
    y = np.zeros(N*K, dtype='uint8')
    for j in xrange(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return X,y

def plot_data(X,y):
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()

def plot_classifier(clf, X, y, xr = [-2,2], yr = [-2,2]):
        """
        Plot the decision boundary of the resulting classifier

        Parameters
        --------------------
            clf   -- classifier
            X     -- numpy array of shape (N,D), samples
            y     -- numpy array of shape (N,), predicted classes
            xr    -- range of x axis
            yr    -- range of y axis

        """

        h = 0.02
        x_min, x_max = xr
        y_min, y_max = yr
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

class LinearClassifier(Classifier) :

    def __init__(self) :
        """
        A linear classifier.

        Attributes
        --------------------
            W -- weight
            b -- bias
        """
        self.W = None
        self.b = None

    def fit(self, X, y, plot_curve = False) :
        """
        Build a linear classifier from the training set (X, y).

        Parameters
        --------------------
            X           -- numpy array of shape (N,D), samples
            y           -- numpy array of shape (N,), target classes
            plot_curve  -- plot learning curve

        Returns
        --------------------
            self -- an instance of self
        """
        N,D = X.shape
        K = np.max(y) + 1
        # initialize parameters randomly
        W = 0.01 * np.random.randn(D,K)
        b = np.zeros((1,K))

        # some hyperparameters
        step_size = 1e-0
        reg = 1e-3 # regularization strength

        # gradient descent loop
        num_examples = X.shape[0]
        losses = []
        for i in xrange(200):

            # evaluate class scores, [N x K]
            scores = np.dot(X, W) + b 

            # compute the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(probs[range(num_examples),y])
            data_loss = np.sum(corect_logprobs)/num_examples
            reg_loss = 0.5*reg*np.sum(W*W)
            loss = data_loss + reg_loss
            losses.append(loss)

            # compute the gradient on scores
            dscores = probs
            dscores[range(num_examples),y] -= 1
            dscores /= num_examples

            # backpropate the gradient to the parameters (W,b)
            dW = np.dot(X.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)

            dW += reg*W # regularization gradient

            # perform a parameter update
            W += -step_size * dW
            b += -step_size * db
        
        self.W = W
        self.b = b

        if plot_curve:
            plt.plot(range(200), losses, 'r', label = 'training loss')
            plt.legend(loc = 'upper right')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Softmax Linear Classifier : Training Loss v.s. Iterations')
            plt.show()
        
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (N,D), samples

        Returns
        --------------------
            y    -- numpy array of shape (N,), predicted classes
        """
        if self.W is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        y = np.dot(X, self.W) + self.b
        y = np.argmax(y, axis=1)
        return y

class NeuralNet(Classifier) :

    def __init__(self,h=100) :
        """
        A linear classifier.

        Attributes
        --------------------
            layer1_parameters -- (weight, bias) of layer 1
            layer2_parameters -- (weight, bias) of layer 2
        """
        self.layer1_parameters = None
        self.layer2_parameters = None
        self.h = h # size of hidden layer

    def fit(self, X, y, plot_curve = False) :
        """
        Build a linear classifier from the training set (X, y).

        Parameters
        --------------------
            X           -- numpy array of shape (N,D), samples
            y           -- numpy array of shape (N,), target classes
            plot_curve  -- plot learning curve

        Returns
        --------------------
            self -- an instance of self
        """
        N,D = X.shape
        K = np.max(y) + 1
        # initialize parameters randomly
        h = self.h
        W = 0.01 * np.random.randn(D,h)
        b = np.zeros((1,h))
        W2 = 0.01 * np.random.randn(h,K)
        b2 = np.zeros((1,K))

        # some hyperparameters
        step_size = 1e-0
        reg = 1e-3 # regularization strength

        # gradient descent loop
        num_examples = X.shape[0]
        losses = []
        for i in xrange(10000):

            # evaluate class scores, [N x K]
            hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
            scores = np.dot(hidden_layer, W2) + b2

            # compute the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(probs[range(num_examples),y])
            data_loss = np.sum(corect_logprobs)/num_examples
            reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
            loss = data_loss + reg_loss
            losses.append(loss)

            # compute the gradient on scores
            dscores = probs
            dscores[range(num_examples),y] -= 1
            dscores /= num_examples

            # backpropate the gradient to the parameters
            # first backprop into parameters W2 and b2
            dW2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            # next backprop into hidden layer
            dhidden = np.dot(dscores, W2.T)
            # backprop the ReLU non-linearity
            dhidden[hidden_layer <= 0] = 0
            # finally into W,b
            dW = np.dot(X.T, dhidden)
            db = np.sum(dhidden, axis=0, keepdims=True)

            # add regularization gradient contribution
            dW2 += reg * W2
            dW += reg * W

            # perform a parameter update
            W += -step_size * dW
            b += -step_size * db
            W2 += -step_size * dW2
            b2 += -step_size * db2
        
        self.layer1_parameters = (W,b)
        self.layer2_parameters = (W2,b2)

        if plot_curve:
            plt.plot(range(10000), losses,'r', label = 'training loss')
            plt.legend(loc = 'upper right')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Neural Network : Training Loss v.s. Iterations')
            plt.show()
        
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (N,D), samples

        Returns
        --------------------
            y    -- numpy array of shape (N,), predicted classes
        """
        if self.layer1_parameters is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        W,b = self.layer1_parameters
        W2,b2 = self.layer2_parameters
        hidden_layer = np.maximum(0, np.dot(X, W) + b)
        y = np.dot(hidden_layer, W2) + b2
        y = np.argmax(y, axis=1)
        return y

def main():
    ### ========== SECTION : START ========== ###
    # Part (a) Visualize the Dataset
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X,y = load_data(N,D,K)
    plot_data(X,y)
    ### ========== SECTION : END ========== ###

    ### ========== SECTION : START ========== ###
    # Part (b-d) Training a Muli-Class Softmax Linear Classifier
    print('=====Softmax Linear Classifier=====')
    clf = LinearClassifier()
    clf.fit(X,y,plot_curve = True)
    y_pred = clf.predict(X)
    print 'Linear Classifier training accuracy: %.2f' % (np.mean(y_pred == y))
    plot_classifier(clf,X, y, xr = [X[:, 0].min() - 1, X[:, 0].max() + 1], yr = [X[:, 1].min() - 1, X[:, 1].max() + 1])
    ### ========== SECTION : END ========== ###

    ### ========== SECTION : START ========== ###
    # Part (b-d) Training a Neural Network
    print('=====Neural Network=====')
    h = 100 # size of hidden layer
    clf = NeuralNet(h = h)
    clf.fit(X,y,plot_curve = True)
    y_pred = clf.predict(X)
    print 'Neural Network training accuracy: %.2f' % (np.mean(y_pred == y))
    plot_classifier(clf,X, y, xr = [X[:, 0].min() - 1, X[:, 0].max() + 1], yr = [X[:, 1].min() - 1, X[:, 1].max() + 1])

if __name__ == "__main__":
    main()