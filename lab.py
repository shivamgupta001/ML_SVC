#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:39:28 2018

@author: shivam
"""

#Importing libraries
import pandas as pd
import numpy as np

#Importing the dataset
Location = 'ML_assignment.csv'
df = pd.read_csv(Location, names= 	['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10',
	 'v11','v12','v13','v14','v15','v16','v17','v18','v19','v20',
	 'v21','v22','v23','v24','v25','v26','v27','v28','v29','v30',
	 'v31','v32','v33','v34','v35','v36','v37','v38','v39','v40',	
	 'v41','v42'])
y  = df['v42'].values
x  = df.iloc[:,:-1].values

#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_tr, X_tst, y_tr, y_tst = train_test_split(x, y, test_size=0.1, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc    = StandardScaler()
X_tr  = sc.fit_transform(X_tr)
X_tst = sc.transform(X_tst)

#Applying PCA
from sklearn.decomposition import PCA
pca   = PCA(n_components=3)
X_tr  = pca.fit_transform(X_tr)
X_tst = pca.transform(X_tst)

#Fitting Kernel SVM to Training Set
from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=0, C=1000)
clf.fit(X_tr, y_tr)

#Predicting the Test Set Results
predicted = clf.predict(X_tst)

#Measuring Accuracy
from sklearn import metrics
print("SVC\n", metrics.accuracy_score(y_tst, predicted))

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tst, predicted)


#End of code--------------------------------------

#Applying K-Fold Cross Validtion
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=clf, X=X_tr, y=y_tr, cv=20, n_jobs = -1)
accuracies.mean()
accuracies.std()

#Applying the grid search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1, 10, 100, 1000], 'kernel': ['linear']},
              {'C':[1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma':[0.5, 0.1, 0.01, 0.001]}]
grid_search= GridSearchCV( estimator = clf,
                          param_grid = parameters,
                          cv = 10,
                          n_jobs = -1)
grid_search = grid_search.fit(X_tr, y_tr)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#visualizing the training set results
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, Y_set = X_tr, y_tr
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop=X_set[:,0].max()+1, step=0.01),
                     np.arange(start=X_set[:,1].min() -1, stop=X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifying (training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

#NOTE:- Below plots work only for n_components = 2 
#visualizing the test set results
X_set, Y_set = X_tst, y_tst
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop=X_set[:,0].max()+1, step=0.01),
                     np.arange(start=X_set[:,1].min() -1, stop=X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifying (test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()