# -*- coding: utf-8 -*-
# Adds google drive
from google.colab import drive
drive.mount('/content/drive')

# Set the working directory
import os
os.chdir("/content/drive/MyDrive/BZAN 554")

#Most Recent#

#use "Loss" on the x-axis and "Optimizer" on the y-axis to center the points because we want to identify dimension of highest variance.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load the hyperparameters and performance metrics data from Excel file
dataset = pd.read_csv('gridlock.csv')

dataset.head()
#Drops all the nas.
dataset.dropna(inplace=True)

# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dummy_opt = pd.get_dummies(dataset['Optimizer'])
dummy_act = pd.get_dummies(dataset['Hidden_Activation'])

dataset = pd.concat([dataset, dummy_opt], axis=1).reindex(dataset.index)
dataset = pd.concat([dataset, dummy_act], axis=1).reindex(dataset.index)
 
# removing the column 'Purchased' from df
# as it is of no use now.
dataset.drop('Optimizer', axis=1, inplace=True)
dataset.drop('Hidden_Activation', axis=1, inplace=True)

def neuron_fixer(act):
  if (act=='90%'):
    return int(90)
  elif (act=='60%'):
    return int(60)
  else:
    return int(30)

dataset['Neuron_Reduction']=dataset['Neuron_Reduction'].apply(neuron_fixer)

dataset.head()

# distributing the dataset into two components X and Y
X = dataset[['Layers','Epochs','Batch_Sizes','Neuron_Reduction', 
            'adagrad',	'adam',	'lr_scheduler_expdec',
            'momentum',	'nesterov',	'plain_SGD',	'rmsprop',	'elu',
            'leaky_relu',	'prelu',	'relu',	'sigmoid',	'tanh']]
y = dataset['Loss']

from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

# Splitting the X and Y into the
# Training set and Testing set
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size = 0.2, random_state = 0)

# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA
 
pca = PCA(n_components = 2)
 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
 
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression To the training set
from sklearn.linear_model import LogisticRegression 
 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set result using
# predict function under LogisticRegression
y_pred = classifier.predict(X_test)

# making confusion matrix between
#  test set of Y and predicted value.
from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_test, y_pred)

# Predicting the training set
# result through scatter plot
from matplotlib.colors import ListedColormap
 
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                     stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                     stop = X_set[:, 1].max() + 1, step = 0.01))
 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
 
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend
 
# show scatter plot
plt.show()