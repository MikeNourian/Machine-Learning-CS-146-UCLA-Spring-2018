
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

'''
This part of the code is for reference
# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy',max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(X, y)
'''

### ========== SECTION : START ========== ###
# Part (a) Load the data
D = load_digits()
X_train, X_test, y_train, y_test = train_test_split(D.data, D.target, test_size=0.1, random_state=0)
### ========== SECTION : END ========== ###

### ========== SECTION : START ========== ###
print('=====Baseline Classifier=====')
# Part (b) Create a baseline decision tree classifier
clf = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
scores = cross_val_score(clf, X_train, y_train, cv =10)
print('Baseline Cross-validaiton error',1-scores.mean())
### ========== SECTION : END ========== ###


### ========== SECTION : START ========== ###
print('=====Bagging Classifier=====')
# Part (c) 1) Define a DT Bagging Classifier
# Part (c) 2) Report the cross_val_score
bagging = BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy',max_features=None),random_state=0)
scores = cross_val_score(bagging, X_train, y_train, cv = 10)
print('Bagging Decision Tree Cross-validaiton error',1-scores.mean())
### ========== SECTION : END ========== ###

### ========== SECTION : START ========== ###
print('=====Random Forest: Hyperparameter Tuning=====')
# Part (d) 1) and 2) loop through range(1,65) to tune the max_features by
# doing 10 fold cross_validation on random forest using Bagging DT
rf_index = 1
score_max = 0
error = []
for n_features in range(1,65):
    bagging = BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy',max_features=n_features),random_state=0)
    temp = cross_val_score(bagging, X_train, y_train, cv = 10)
    if(temp.mean()>score_max):
        scores = temp
        rf_index = n_features
        score_max = temp.mean()
    error.append(1-temp.mean())
    print n_features,temp.mean()

plt.plot(range(1,65),error,'r', label = 'Cross-validation Error')
plt.legend(loc='upper right')
plt.ylabel('Error')
plt.xlabel('Max Features')
plt.title('Validation Error vs Features')
plt.show()
print('Optimal number of features',rf_index)
print('Random Forest Cross-validationscore',1-scores.mean())
### ========== SECTION : END ========== ###

### ========== SECTION : START ========== ###
print('=====Boosting: Max Iterations Tuning=====')
# Part (e)1) and 2) Perform Validation to find the optimal number of weak learners
# for a good boosted decision stump
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
num =13
# iters = [2**temp for temp in range(0,num)]

iters = [1,2,4,8,16,32,64,128,256,512,1024,1536,2048]
train_error = []
val_error = []

for iter in iters:

    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=3),
                             algorithm="SAMME",
                             n_estimators=iter)

    bdt.fit(X_train1, y_train1)
    train_error.append(1 - accuracy_score(bdt.predict(X_train1), y_train1, normalize=True))
    val_error.append(1 - accuracy_score(bdt.predict(X_val), y_val, normalize=True))
    print iter, val_error[-1]

plt.plot(range(0,len(iters)),train_error,'b',label = 'Train Error')
plt.plot(range(0,len(iters)),val_error,'r', label = 'Validation Error')
plt.legend(loc='upper right')
plt.ylabel('Error')
plt.xlabel('Iterations (log 2 scale)')
plt.title('Training and Validation Error')
plt.show()
print("Minimum validation error",min(val_error))
bo_index = val_error.index(min(val_error))
print("Optimal number of iterations",iters[bo_index])
### ========== SECTION : END ========== ###

### ========== SECTION : START ========== ###
print('===============')
# Part (f) Evaluate the Classifiers on the untouched test set by setting
# the hyperparameters to the tuned values
bagging = BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy',max_features=None),random_state=0)
bagging.fit(X_train,y_train)
print("Bagging test error",1-accuracy_score(bagging.predict(X_test), y_test, normalize=True))

rf_bagging = BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy',max_features=rf_index),random_state=0)
rf_bagging.fit(X_train,y_train)
print("Random forest test error",1-accuracy_score(rf_bagging.predict(X_test), y_test, normalize=True))

bdt = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=3),
                         algorithm="SAMME",
                         n_estimators=iters[bo_index])
bdt.fit(X_train,y_train)
print("Boosted tree test error",1-accuracy_score(bdt.predict(X_test), y_test, normalize=True))
### ========== SECTION : END ========== ###
