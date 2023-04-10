# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:12:43 2018

@author: Sila
"""

import numpy as np
#Import svm model
from sklearn import svm
from sklearn.datasets import load_iris
iris_dataset = load_iris()
#Import svm model
from sklearn import svm

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# random_state simply sets a seed to the random generator,
# so that your train-test splits are always deterministic.
# If you don't set a seed, it is different each time.

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, Y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Prepare the classifier
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

# Fit the model
clf.fit(X_train, Y_train)

# Evaluate the model
print("Training scores: {:.2f}".format(clf.score(X_train, Y_train)))
print("Test scores: {:.2f}".format(clf.score(X_test,Y_test)))

#Training scores: 0.95
#Test scores: 0.87

#Not bad! But 87% is still not 100 %...

#The training is so good that there may be overfitting.