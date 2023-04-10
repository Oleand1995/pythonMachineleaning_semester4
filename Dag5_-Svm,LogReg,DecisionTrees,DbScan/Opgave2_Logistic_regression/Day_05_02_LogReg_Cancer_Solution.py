# -*- coding: utf-8 -*-
"""
Created on Mon September 19 10:02:23 2022

@author: sila
"""

#Import scikit-learn dataset library
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

#Load dataset
cancer = datasets.load_breast_cancer()

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

'''Features:  ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
Labels:  ['malignant' 'benign']'''

# print data(feature)shape
print(cancer.data.shape)

# The top 5 records of the feature set.
# print the cancer data features (top 5 records)
print(cancer.data[0:5])

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.80,random_state=109) # 70% training and 30% test


logreg = LogisticRegression(C=1, max_iter=10000, penalty='l1', random_state=None, solver='liblinear',)

# Fit the model
logreg.fit(X_train, y_train)

# Evaluate the model
print("Training scores: {:.2f}".format(logreg.score(X_train, y_train)))
print("Test scores: {:.2f}".format(logreg.score(X_test,y_test)))