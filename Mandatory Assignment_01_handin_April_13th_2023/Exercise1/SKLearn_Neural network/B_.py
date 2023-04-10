# -*- coding: utf-8 -*-
"""
Created on Mon December 9 15:16:37 2018

@author: sila
"""

from sklearn.datasets import make_moons;
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

from matplotlib.colors import ListedColormap, colorConverter

cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
cm2 = ListedColormap(['#0000aa', '#ff2020'])

from matplotlib import pyplot
from pandas import DataFrame

def plot_2d_separator(classifier, X, fill=False, ax=None, eps=None, alpha=1,
                      cm=cm2, linewidth=None, threshold=None,
                      linestyle="solid"):
    # binary?
    if eps is None:
        eps = X.std() / 2.

    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    try:
        decision_values = classifier.decision_function(X_grid)
        levels = [0] if threshold is None else [threshold]
        fill_levels = [decision_values.min()] + levels + [
            decision_values.max()]
    except AttributeError:
        # no decision_function
        decision_values = classifier.predict_proba(X_grid)[:, 1]
        levels = [.5] if threshold is None else [threshold]
        fill_levels = [0] + levels + [1]
    if fill:
        ax.contourf(X1, X2, decision_values.reshape(X1.shape),
                    levels=fill_levels, alpha=alpha, cmap=cm)
    else:
        ax.contour(X1, X2, decision_values.reshape(X1.shape), levels=levels,
                   colors="black", alpha=alpha, linewidths=linewidth,
                   linestyles=linestyle, zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())


def discrete_scatter(x1, x2, y=None, markers=None, s=10, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=None):
    """Adaption of matplotlib.pyplot.scatter to plot classes or clusters.
    Parameters
    ----------
    x1 : nd-array
        input data, first axis
    x2 : nd-array
        input data, second axis
    y : nd-array
        input data, discrete labels
    cmap : colormap
        Colormap to use.
    markers : list of string
        List of markers to use, or None (which defaults to 'o').
    s : int or float
        Size of the marker
    padding : float
        Fraction of the dataset range to use for padding the axes.
    alpha : float
        Alpha value for all points.
    """
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))

    unique_y = np.unique(y)

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []

    current_cycler = mpl.rcParams['axes.prop_cycle']

    for i, (yy, cycle) in enumerate(zip(unique_y, current_cycler())):
        mask = y == yy
        # if c is none, use color cycle
        if c is None:
            color = cycle['color']
        elif len(c) > 1:
            color = c[i]
        else:
            color = c
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .4:
            markeredgecolor = "grey"
        else:
            markeredgecolor = "black"

        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s,
                             label=labels[i], alpha=alpha, c=color,
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])

    if padding != 0:
        pad1 = x1.std() * padding
        pad2 = x2.std() * padding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(x1.min() - pad1, xlim[0]), max(x1.max() + pad1, xlim[1]))
        ax.set_ylim(min(x2.min() - pad2, ylim[0]), max(x2.max() + pad2, ylim[1]))

    return lines


# generate 2d classification dataset
X, y = make_moons(n_samples=100, noise=0.1)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

pyplot.show()


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[8, 6, 4], alpha=0.1)
mlp.fit(X_train, y_train)

plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

pyplot.show()



#Spørgsmål B Classifikation.
#Neuronet score.
print("Neural net: ")
print("Training scores: {:.2f}".format(mlp.score(X_train, y_train)))
print("Test scores: {:.2f}".format(mlp.score(X_test,y_test)))


# Logistisk Reg.
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

# Fit the model
logreg.fit(X_train, y_train)

# Evaluate the model
print("Logreg: ")
print("Training scores: {:.2f}".format(logreg.score(X_train, y_train)))
print("Test scores: {:.2f}".format(logreg.score(X_test,y_test)))
print()

#Viser model i plot.
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

# Plot the decision boundary for the logistic regression model
plt.figure(figsize=(8, 6))
plot_decision_boundary(logreg, X_train, y_train)
plt.title("Logistic Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


#K-Means
from sklearn.cluster import KMeans # This will be used for the algorithm

k = 2
#running kmeans clustering into two
kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train, y_train)
print("Kmeans:")
print("Training scores: {:.2f}".format(kmeans.score(X_train, y_train)))
print("Test scores: {:.2f}".format(kmeans.score(X_test,y_test)))
print()

#Viser model i plot
def plot_clusters(model, X):
    labels = model.labels_
    centers = model.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)

# Plot the clusters for the K-means model
plt.figure(figsize=(8, 6))
plot_clusters(kmeans, X_train)
plt.title("K-means Cluster")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


#Decisions trees
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2) ## indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(X_train, y_train) # training the classifier

# Evaluate the model
print("Decision Tree:")
print("Training scores: {:.2f}".format(tree_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(tree_clf.score(X_test,y_test)))
print()

#Viser model.
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 6))
plot_tree(tree_clf, filled=True, feature_names=['X', 'Y'])
plt.show()


#Randomforrest
from sklearn.ensemble import RandomForestClassifier
random_clf = RandomForestClassifier(n_estimators=20)  # using 20 trees
random_clf.fit(X_train, y_train)

# Evaluate the model
print("Random Forest:")
print("Training scores: {:.2f}".format(random_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(random_clf.score(X_test,y_test)))
print()

#Viser model.
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Plot decision boundary
plot_decision_regions(X_test, y_test, clf=random_clf, legend=2)

# Add axes labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Forest')

# Show the plot
plt.show()


#SVM Linaer.
from sklearn import svm
#Create a svm Classifier
svm_clf = svm.SVC(kernel='linear')  # Linear Kernel.
#Train the model using the training sets
svm_clf.fit(X_train, y_train)

# Evaluate the model
print("SVM Linear: ")
print("Training scores: {:.2f}".format(svm_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(svm_clf.score(X_test,y_test)))
print()

#viser model.
# Get the support vectors and decision function
support_vectors = svm_clf.support_vectors_
decision_function = svm_clf.decision_function(X_train)

# Plot the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolor='k')

# Plot the support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='gray', s=100, marker='o', linewidths=1)

# Create a grid of points in the feature space
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min()-1, X_train[:, 0].max()+1, 500),
                     np.linspace(X_train[:, 1].min()-1, X_train[:, 1].max()+1, 500))
X_grid = np.column_stack((xx.ravel(), yy.ravel()))

# Compute the decision boundary
Z = svm_clf.predict(X_grid)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.2)

# Set the plot axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM Linaer')

# Show the plot
plt.show()


#SVM polynomial
from sklearn import svm
#Create a svm Classifier
svm_clf = svm.SVC(kernel='poly')  # polynomial Kernel.
#Train the model using the training sets
svm_clf.fit(X_train, y_train)

# Evaluate the model
print("SVM polynomial: ")
print("Training scores: {:.2f}".format(svm_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(svm_clf.score(X_test,y_test)))
print()

#viser model.
# Get the support vectors and decision function
support_vectors = svm_clf.support_vectors_
decision_function = svm_clf.decision_function(X_train)

# Plot the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolor='k')

# Plot the support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='gray', s=100, marker='o', linewidths=1)

# Create a grid of points in the feature space
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min()-1, X_train[:, 0].max()+1, 500),
                     np.linspace(X_train[:, 1].min()-1, X_train[:, 1].max()+1, 500))
X_grid = np.column_stack((xx.ravel(), yy.ravel()))

# Compute the decision boundary
Z = svm_clf.predict(X_grid)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.2)

# Set the plot axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM polynomial')

# Show the plot
plt.show()


#SVM rbf
from sklearn import svm
#Create a svm Classifier
svm_clf = svm.SVC(kernel='rbf')  # Linear Kernel, Kernel skal prøves med forskellige parametre
#Train the model using the training sets
svm_clf.fit(X_train, y_train)

# Evaluate the model
print("SVM rbf: ")
print("Training scores: {:.2f}".format(svm_clf.score(X_train, y_train)))
print("Test scores: {:.2f}".format(svm_clf.score(X_test,y_test)))
print()

#viser model.
# Get the support vectors and decision function
support_vectors = svm_clf.support_vectors_
decision_function = svm_clf.decision_function(X_train)

# Plot the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolor='k')

# Plot the support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='gray', s=100, marker='o', linewidths=1)

# Create a grid of points in the feature space
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min()-1, X_train[:, 0].max()+1, 500),
                     np.linspace(X_train[:, 1].min()-1, X_train[:, 1].max()+1, 500))
X_grid = np.column_stack((xx.ravel(), yy.ravel()))

# Compute the decision boundary
Z = svm_clf.predict(X_grid)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.2)

# Set the plot axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM rbf')

# Show the plot
plt.show()












