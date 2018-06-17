# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# classes
######################################################################

class Data :

    def __init__(self, X=None, y=None) :
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname('__file__')
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

    def plot(self, **kwargs) :
        """Plot data."""

        if 'color' not in kwargs :
            kwargs['color'] = 'b'

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :

    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.

        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param


    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].

        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features

        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n,d = X.shape

        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        Phi = X
        m = self.m_

        if m == 1:
            Phi = np.zeros((n,2))
            for i in range(n):
                Phi[i,0] = 1
                Phi[i, 1] = X[i]

        else:
            Phi = np.ones((n,m+1))#n*m+1 dimmension
            power_arr = np.arange(0, m+1)
            for index, row in enumerate(Phi):# get every row
                row = np.repeat(X[index],m+1)
                row = np.power(row,power_arr)
                Phi [index,] = row
        #also could use the following
        """
        import sklearn.preprocessing as sk
        #X is a N*1 vector
        poly_mat = sk.PolynomialFeatures(3)
        poly.fit_transform(a)
        """





        ### ========== TODO : END ========== ###

        return Phi


    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent. full batch

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes

        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")

        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration, the value of error on each iteration



        # GD loop
        for t in xrange(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            #print ("this eta line was exectued")
            eta = 1/float(1+t)
            #print eta
            #eta = eta_input
            #print ("eta value is %f, and t = %d") % (eta,t)




            ### ========== TODO : END ========== ###

            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math

            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            # IMPORTANT: The reason you can not use predict or cost is because they operat
            # delta_J = X_T*X*Theta - X_T*y

            #updated
            y_pred_vector = np.dot(X, self.coef_)
            delta_J = np.dot(X.transpose(),np.dot(X,self.coef_)) - np.dot(X.transpose(),y)


            #cost = np.dot((y - y_pred_vector).transpose(), (y - y_pred_vector))  # write in the matrix form

            #updated
            #self.coef_ = self.coef_ - eta*delta_J
            #update the coeffient here
            #for x in X, y in y: #returns row
            for index, x in enumerate(X):
                #get prediction
                #w.T*X
                y_true = y[index]
                pred = np.dot(x, self.coef_)
                #make sure that the values are compatible
                self.coef_ = self.coef_ - 2*eta *(pred - y_true)*x


            y_pred = np.dot(X, self.coef_) # change this line
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)
            ### ========== TODO : END ========== ###

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break

            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print 'number of iterations: %d' % (t+1)

        return self, t


    def fit(self, X, y, l2regularize = None ) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization

        Returns
        --------------------
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        X_X_T = np.linalg.pinv(np.dot(X.transpose(),X) + l2regularize*np.identity(np.shape(X.transpose())[0]))
        self.coef_ = np.dot(X_X_T,np.dot(X.transpose(),y))


        ### ========== TODO : END ========== ###

        return self


    def predict(self, X) :
        """
        Predict output for X.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features

        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part c: predict y
        # for this we first get the single value of feature vector, then X in the transposed form and then we have to multiply by Theta

        y = np.dot(X, self.coef_)#coef is the coef matrix
        ### ========== TODO : END ========== ###


        return y


    def cost(self, X, y) :
        """
        Calculates the objective function.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        #we know for linear/polynomial regression, the cost is the square of the errors
        X = self.generate_polynomial_features(X)
        y_pred_vector = np.dot(X, self.coef_)
        cost = np.dot((y-y_pred_vector).transpose(),(y-y_pred_vector))#write in the matrix form
        ### ========== TODO : END ========== ###
        return cost


    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        n, d = X.shape
        error = np.sqrt(self.cost(X,y)/n)
        ### ========== TODO : END ========== ###
        return error


    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()




######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')
    X = train_data.X
    y = train_data.y
    X_test = test_data.X
    y_test = test_data.y

    """
    #test code
    train_data = load_data('regression_train.csv')
    model = PolynomialRegression()
    model.coef_ = np.zeros(2)
    cost = model.cost(train_data.X, train_data.y)
    print "cost is = ", cost
    
    """

#begin of the comment

    """
    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print 'Visualizing data...'
    plt.scatter(X,y)
    plt.xlabel("X, features")
    plt.ylabel("Y, labels")
    plt.title("Scatter Plot of Features vs Labels for Training Samples")
    plt.show()

    """






    ### ========== TODO : END ========== ###


    #commented linear regression

    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print 'Investigating linear regression...'
    model = PolynomialRegression(m=1)
 #   X_Mat = model.generate_polynomial_features(X)

    #steps = [10**(-4), 10**(-3), 10**(-2), 0.0407]

    steps = [1] #we dont care bout this really for the case of changing the step size
    print ": Coefficients : Number of iterations : Final Objective Function :"
    for step in steps:
        model = PolynomialRegression(m=1)#linear regression
        model, iter_count = model.fit_GD(X,y,eta=step)
        print model.coef_, iter_count+1, model.cost(X,y) #final value of target function
    #make a table of the coefficients, number of iterations until convergence (this number will be 10, 000 if the algorithm did not converge in a smaller number of iterations) and the final value of the objective function.

    #commented here
    """
    model = PolynomialRegression(m = 1)
    model = model.fit(X,y)
    print "the closed theta value is"
    print model.coef_, model.cost(X, y)
    """
    #to here



    ### ========== TODO : END ========== ###

    """
    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print 'Investigating polynomial regression...'

    rms_train = []
    rms_test = []
    m_array = np.arange(0, 11)
    for m in m_array:
        model = PolynomialRegression(m=m)
        model = model.fit(X,y)
        train_rms = model.rms_error(X,y)
        rms_train.append(train_rms)

        test_rms = model.rms_error(X_test, y_test)
        rms_test.append(test_rms)

    #now draw the plot
    plt.plot(m_array, rms_train,'b',label= "Train RMSE")
    plt.plot(m_array, rms_test, 'g', label= "Test RMSE")
    plt.xlabel("m, Order of Polynomial")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    ### ========== TODO : END ========== ###
    """

    #begin of comment
    """
    ### ========== TODO : START ========== ###
    # parts j-k: main code for regularized regression
    print 'Investigating regularized regression...'
    exponents = []
    exponents.append(np.arange(0,-9,-1))
    repeat = np.repeat(10.0,9)
    lambda_arr = np.append(np.array([0]),np.power(repeat,exponents))

    train_rms = []
    test_rms = []
    #train a model with fit and explicit solution
    for lam in lambda_arr:
        model = PolynomialRegression(m=10)
        model.fit(X,y,l2regularize=lam)

        #calculate RMS error for both train and test
        rms = model.rms_error(X,y)
        train_rms.append(rms)

        rms = model.rms_error(X_test,y_test)
        test_rms.append(rms)

    #now plot of x[0,...,10]
    x = range(1,11)
    plt.plot(x, train_rms, 'b', label = "Train RMSE")
    plt.plot(x, test_rms, 'g', label="Test RMSE")
    plt.legend()
    plt.xlabel("Lambda, Regularization Parameter")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Lambda (Regularization Parameter")
    plt.show()

    ### ========== TODO : END ========== ###
    
    """
    #end of comment
    print "Done!"

if __name__ == "__main__" :
    main()
