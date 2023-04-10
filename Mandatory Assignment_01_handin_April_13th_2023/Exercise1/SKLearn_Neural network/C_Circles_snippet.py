# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:23:02 2018

@author: sila
"""

from sklearn.datasets import make_circles;
import matplotlib.pyplot as plt

from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()


#OPG C.
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
from sklearn import svm


#SVM rbf/Gaussian.
#Create a svm Classifier
svm_clf = svm.SVC(kernel='rbf')  # Linear Kernel.
#Train the model using the training sets
svm_clf.fit(X_train, y_train)

# Evaluate the model
print("SVM rbf: ")
print("Training scores: {:.2f}".format(svm_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(svm_clf.score(X_test,y_test)))
print()

#viser model.
# Define a meshgrid of points for plotting the decision boundaries
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 500), np.linspace(-1.5, 1.5, 500))
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries and the scatter plot of the data
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM rpf Decision Boundaries')
plt.show()

# Show the plot
plt.show()


#SVM Sigmoid.
#Create a svm Classifier
svm_clf = svm.SVC(kernel='sigmoid')  # Linear Kernel.
#Train the model using the training sets
svm_clf.fit(X_train, y_train)

# Evaluate the model
print("SVM sigmoid: ")
print("Training scores: {:.2f}".format(svm_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(svm_clf.score(X_test,y_test)))
print()

#viser model.
# Define a meshgrid of points for plotting the decision boundaries
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 500), np.linspace(-1.5, 1.5, 500))
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries and the scatter plot of the data
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM sigmoid Decision Boundaries')
plt.show()

# Show the plot
plt.show()


#SVM Sigmoid.
#Create a svm Classifier
svm_clf = svm.SVC(kernel='poly')  # Linear Kernel.
#Train the model using the training sets
svm_clf.fit(X_train, y_train)

# Evaluate the model
print("SVM poly: ")
print("Training scores: {:.2f}".format(svm_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(svm_clf.score(X_test,y_test)))
print()

#viser model.
# Define a meshgrid of points for plotting the decision boundaries
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 500), np.linspace(-1.5, 1.5, 500))
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries and the scatter plot of the data
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM poly Decision Boundaries')
plt.show()

# Show the plot
plt.show()







