#Titanic dataset predictions

#import panda library and a few others we will need.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
# skipping the header
data =pd.read_csv( 'titanic_train_500_age_passengerclass.csv' , sep = ',' , header = 0 )

# show the data
print ( data .describe( include = 'all' ))
#the describe is a great way to get an overview of the data
print ( data .values)

# Replace unknown values. Unknown class set to 3
data["Pclass"].fillna(3, inplace = True)

# Replace unknown values. Unknown age set to 25
data["Age"].fillna(27, inplace = True)

# Replace unknown values. Unknown survival set to survived
data["Survived"].fillna(1, inplace = True)


yvalues = pd.DataFrame( dict ( Survived =[]), dtype = int )
yvalues[ "Survived" ] = data [ "Survived" ].copy()
#now the yvalues should contain just the survived column

x = data[ "Age" ]
y = data[ "Pclass" ]
plt.figure("X-Alder / Y-Klasse")
plt.scatter(x.values, y.values, color = 'black' , s = 20 )

plt.show()

#now we can delete the survived column from the data (because
#we have copied that already into the yvalues.
data.drop( 'Survived' , axis = 1 , inplace = True )

data.drop( 'PassengerId' , axis = 1 , inplace = True )

# show the data
print ( data.describe( include = 'all' ))

xtrain = data.head( 400 )
xtest = data.tail( 100 )

ytrain = yvalues.head( 400 )
ytest = yvalues.tail( 100 )

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
mlp = MLPClassifier( hidden_layer_sizes =(8,4), max_iter = 1000,
random_state = 0 )

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
print("Values from matrix: ",tn,fp,fn,tp)
