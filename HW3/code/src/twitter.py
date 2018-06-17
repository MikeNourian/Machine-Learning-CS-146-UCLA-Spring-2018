


from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    index = 0
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        for line in fid:
            words = extract_words(line)
            for word in words:
                if word_list.__contains__(word) == False:
                    word_list[word] = index
                    index = index + 1
        ### ========== TODO : END ========== ###
    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    #so the feature matrix is vertical
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        #start here
        #add the vectors with height and width specified

        for index, line in enumerate(fid):
            words_extracted = extract_words(line)
            for word in words_extracted:
                dic_index = word_list[word]
                feature_matrix[index,dic_index] = 1



        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 #set the 0 values of the label to 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    score = 0
    if metric == 'accuracy':
        score = metrics.accuracy_score(y_true,y_label)
    elif metric == 'f1_score':
        score = metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        score = metrics.roc_auc_score(y_true,y_label)
    elif metric == 'precision':
        score = metrics.precision_score(y_true, y_label)

    # TPR is also known as sensitivity
    # FPR is one minus the specificity or true negative rate
    elif metric == 'sensitivity':
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        #TP/(TP+FN)
        score = conf_matrix[1,1]/float((conf_matrix[1,1]+conf_matrix[1, 0]))
    elif metric == 'specificity':
        #TN/TN+FP
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        score = conf_matrix[0,0]/float((conf_matrix[0,0]+conf_matrix[0,1]))
    else:
        print "something was wrong in performance function"



    return score
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"): #kf in this case
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance
    #we should try to keep the positives and negatives the same proportion
    #remember that we can have 3 parts, train, dev, and test but that wastes a lot of space
    #so use cross validation and train on k-1 data, test on the rest of the data

    #kf that is already initialized
    score = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #train on the X_train and test using the y value
        #at the end

        clf.fit(X_train, y_train)#train the svm model
        y_pred = clf.decision_function(X_test) #we can just use the decision_function as we take the sign in the performance function
        perf = performance(y_test,y_pred,metric)
        score.append(perf)



    #take the average of the scores
    return np.mean(score)
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """



    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)


    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    scores = []
    for c in C_range:
        clf = SVC(C=c,kernel="linear") #we want to find the best c value
        score = cv_performance(clf,X,y,kf,metric = metric)
        scores.append(score)
    print("Scores are, ", scores)
    max_index = scores.index(max(scores))
    return C_range[max_index]
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation\
    #initialize the classfier with rbf = kernel and different c and gamma values

    #IMPORTANT: scoring is where I pass in the function performance_cv

    #Using sklearn.GridSearchCV it was seen that the performance is best for c = 10, gamma = 0.01 based on the metric accuracy
    #Added
    """
    param_grid = dict(gamma=gamma_range, C=C_range)
    #cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    cv = StratifiedShuffleSplit(y=y,test_size=0.2, n_iter=5,random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    """
    #End of added

    C_range = np.logspace(-2, 2, 5)
    gamma_range = np.logspace(-2, 2, 5)
    max_score = 0
    best_c = 0
    best_gamma = 0
    for c in C_range:
        for gamma in gamma_range:
            score = cv_performance(SVC(C=c,kernel='rbf',gamma=gamma),X=X, y=y,kf=kf,metric=metric)
            if score > max_score:
                max_score = score
                best_c = c
                best_gamma = gamma

    print "max score for ", metric, "= ", max_score
    return best_gamma, best_c
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    #first get y_pred
    y_pred = clf.decision_function(X)
    score = performance(y_true=y, y_pred=y_pred, metric= metric)

    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size= 0.111)

    # part 2b: create stratified folds (5-fold CV)
    skf = StratifiedKFold(y_train,n_folds=5) #the cross validation should be used on the training data
    #Start: for linear

    max_score_arr = {}
    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    for metric in metric_list:
        c_max = select_param_linear(X_train,y_train,skf,metric=metric)
        max_score_arr[metric] = c_max

    print "hello"

    #end for linear
    #Begin of rbf
    """
    gamma_c_dict = {}
    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV

    for metric in metric_list:
        best_gamma, best_c = select_param_rbf(X_train,y_train,kf=skf,metric=metric)
        gamma_c_dict[metric] = best_gamma, best_c
    print 'hello'
    """
    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    """
    Based on the results you obtained in Section 0.2 and Section 0.3, choose a hyperpa- rameter setting for the linear-kernel SVM and a hyperparameter 
    setting for the RBF-kernel SVM. Explain your choice.
    Then, in main(...), using the training data extracted in Section 0.1 and SVC.fit(...), train a linear- and an RBF-kernel SVM with your chosen settings.
    """
    #Started here
    """
    
    #create the model
    linear_SVM = SVC(C=1.0, kernel='linear')
    rbf_SVM = SVC(C=10.0, gamma=0.01, kernel='rbf')
    #train the model
    linear_SVM.fit(X_train, y_train)
    rbf_SVM.fit(X_train, y_train)

    linear_score_dictionary = {}
    rbf_score_dictionary = {}
    # part 4c: report performance on test data
    for metric in metric_list:
        linear_score = performance_test(linear_SVM,X_test, y= y_test, metric = metric)
        rbf_score = performance_test(rbf_SVM,X_test,y_test,metric)
        linear_score_dictionary[metric] = linear_score
        rbf_score_dictionary[metric] = rbf_score

    for key, v in rbf_score_dictionary.iteritems():
        print ("k= ", key, "value= ", v)

    ### ========== TODO : END ========== ###
    """
    #Ended here
    
if __name__ == "__main__" :
    main()
