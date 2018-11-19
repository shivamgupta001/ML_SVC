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
df = pd.read_csv(Location, names=
 ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10',
  'v11','v12','v13','v14','v15','v16','v17','v18','v19','v20',
  'v21','v22','v23','v24','v25','v26','v27','v28','v29','v30',
  'v31','v32','v33','v34','v35','v36','v37','v38','v39','v40','v41','v42'])
y = df['v42'].values
x = df.iloc[:,:-1].values

#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_tr, X_tst, y_tr, y_tst = train_test_split(x, y, test_size=0.2, random_state=0)



# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_tst = sc.fit_transform(X_tst)

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_tr = pca.fit_transform(X_tr)
X_tst = pca.fit_transform(X_tst)
explained_variance = pca.explained_variance_ratio_


#SVC
from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=0)
clf.fit(X_tr, y_tr)
predicted = clf.predict(X_tst)

#Measuring Accuracy
from sklearn import metrics
print("SVC\n", metrics.accuracy_score(y_tst, predicted))

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tst, predicted)

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