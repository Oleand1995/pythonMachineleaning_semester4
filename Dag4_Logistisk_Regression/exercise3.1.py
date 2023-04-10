import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#Indsætter points i tabellen.
# Array of points with a classification

#Forskellige set af points.
#X = np.array(np.matrix('6,350; 2.5, 400;4.5,500; 3.5,350; 2,300;4,600;7,300;5,500;5,400;6,400;3,400;4,500;1,200;3,400;7,700;3,550;2.5,650'))
#y = np.array(np.matrix('0;0;1;0;0;1;1;1;0;1;0;0;0;0;1;1;0'))[:,0]
#X = np.array(np.matrix('4,450;5,600;6,700;4.5,550;4.9,500;5,650;5.5,500; 5.25,525; 4.25,625; 4.75,575'))
#y = np.array(np.matrix('0;1;1;0;0;1;0;1;1;1'))[:,0]

#Her laver jeg et dataset, med banklån. her må man låne 2000 hvis man tjener fx 5000.
X = np.array(np.matrix('5000,2000;8000,8000;10000,5000;10000,8000;12000,2000;17000,8000;8763,17000'))
y = np.array(np.matrix('1;0;1;0;1;1;0'))[:,0]

pos=np.where(y==1)
neg=np.where(y==0)
plt.plot(X[pos[0],0], X[pos[0],1], 'ro')
plt.plot(X[neg[0],0], X[neg[0],1], 'bo')
plt.xlim([min(X[:,0]),max(X[:,0])])
plt.ylim([min(X[:,1]),max(X[:,1])])
plt.plot


#Her indsætte en linær regression.
logreg = linear_model.LogisticRegression(C=1000)
model = logreg.fit(X, y)
# model.coef_[0,0]*x + model.coef_[0,1]*y + model.intercept_[0] = 0
# y = - ( model.coef_[0,0]*x + model.intercept_[0]) / model.coef_[0,1]
xx = np.linspace(1, 100000)
yy = - (model.coef_[0,0] / model.coef_[0,1]) * xx - ( model.intercept_[0] / model.coef_[0,1])
plt.plot(xx, yy, 'k-')


plt.show()