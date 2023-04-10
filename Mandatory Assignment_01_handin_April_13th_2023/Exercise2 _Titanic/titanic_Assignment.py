#Titanic dataset predictions

#import panda library and a few others we will need.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
# skipping the header
data =pd.read_csv( 'titanic_800.csv' , sep = ',' , header = 0 )

# show the data
print ( data .describe( include = 'all' ))
#the describe is a great way to get an overview of the data
print ( data .values)

# Replace unknown values. Unknown class set to 3
data["Pclass"].fillna(3, inplace = True)

# Replace unknown values. Unknown age set to 27
data["Age"].fillna(27, inplace = True)

# Replace unknown values. Unknown survival set to survived
data["Survived"].fillna(1, inplace = True)

#Sætter embarked.
data['Embarked'].fillna(0, inplace = True)


yvalues = pd.DataFrame( dict ( Survived =[]), dtype = int )
yvalues[ "Survived" ] = data [ "Survived" ].copy()
#now the yvalues should contain just the survived column


#now we can delete the survived column from the data (because
#we have copied that already into the yvalues.
data.drop( 'Survived' , axis = 1 , inplace = True )

data.drop( 'PassengerId' , axis = 1 , inplace = True )

data.drop( 'Name' , axis = 1 , inplace = True )

data.drop( 'Cabin' , axis = 1 , inplace = True )

data.drop( 'Ticket' , axis = 1 , inplace = True )

#converter.
data['Sex'] = data['Sex'].replace(['female'],1 )
data['Sex'] = data['Sex'].replace(['male'],0 )

data['Embarked'] = data['Embarked'].replace(['S'],0 )
data['Embarked'] = data['Embarked'].replace(['C'],1 )
data['Embarked'] = data['Embarked'].replace(['Q'],2 )


#Laver scatter plots.
x = data[ "Age" ]
y = data[ "Pclass" ]
plt.figure("X-Alder / Y-Klasse")
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
plt.show()

x = data[ "Age" ]
y = data[ "Sex" ]
plt.figure("X-Alder / Y-Klasse")
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
plt.show()


x = data[ "Age" ]
y = data[ "Embarked" ]
plt.figure("X-Alder / Y-Klasse")
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
plt.show()


# show the data
print ( data.describe( include = 'all' ))

#træner på 80% og tester på 20%
xtrain = data.head( 640 )
xtest = data.tail( 160 )

ytrain = yvalues.head( 640 )
ytest = yvalues.tail( 160 )

print ( ytrain )

#Mit arbejde her fra.
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#But first we need to scale our data - there is a normal requirement for working with
#neural networks.
scaler = StandardScaler()
scaler .fit(xtrain)
xtrain = scaler .transform(xtrain)
xtest= scaler .transform(xtest)

#Here you can see an example for defining with 1000 epochs for a neural network
#with 2 hidden layers, where each hidden layer has 8 neurons:
mlp = MLPClassifier( hidden_layer_sizes =(8,6,4), max_iter = 1000,random_state = 0, alpha=0.1 )

#Scikit-learn will automatically add the correct number of input neurons and the
#correct number of neurons in the output layer based on the xtrain and ytrain data.
#You will then need to train the model to the data:
mlp.fit(xtrain,ytrain.values.ravel())
#the reason for the values.ravel is that these data has not
#been scaled and they need to be converted to the correct
#input format for the mlp.fit. Data that is scaled already has
#this done to them.

#to predict on our xtest set
predictions = mlp.predict(xtest)

#Now is the time to evaluate how good our predictor is:
matrix = confusion_matrix(ytest,predictions)
print ("Matrix: \n",matrix)
print ("classification_report: \n",classification_report(ytest,predictions))

#You can extract the values from the matrix like this:
tn, fp, fn, tp = matrix.ravel()

#Print values from matrix.
print("Values from matrix: ",tp,fp,fn,tn)

#Decisions trees
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=10) ## indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(xtrain, ytrain) # training the classifier

# Evaluate the model
print("Decision Tree:")
print("Training scores: {:.2f}".format(tree_clf.score(xtrain, ytrain)))
print("Test scores: {:.2f}".format(tree_clf.score(xtest,ytest)))
print()

predictions_tree = tree_clf.predict(xtest)

matrix_tree = confusion_matrix(ytest,predictions_tree)

print ("Matrix_Tree: \n",matrix_tree)
print ("classification_report: \n",classification_report(ytest,predictions_tree))
