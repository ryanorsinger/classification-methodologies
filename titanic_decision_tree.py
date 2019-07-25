# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from acquire import get_titanic_data
from prepare import prepare_titanic_data

df = prepare_titanic_data(get_titanic_data())

df.head()
# Set Feature variables and target variable
X = df[["sex_encoded", "pclass", "sibsp", "parch", "fare", "embarked", "class_encoded", "alone"]]
y = df[["survived"]]

# Split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)
X_train.head()

# for classification you can change the algorithm as gini or entropy (information gain).  Default is gini.
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)

# Fit the model to the training data
clf = clf.fit(X_train, y_train)

# Estimate survival predictions
y_pred = clf.predict(X_train)

# Estimate the probability of survival
y_pred_proba = clf.predict_proba(X_train)

# Evaluate Model
# Accuracy: number of correct predictions over the number of total instances that have been evaluated.
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

# Produce confusion matrix
labels = sorted(y_train.survived.unique())

# Label the rows and columns for the confusion matrix
columns = ["Predicted Death", "Predicted Survival"]
rows = ["Actual Death", "Actual Survival"]
print(pd.DataFrame(confusion_matrix(y_train, y_pred), index=rows, columns=columns))
print()
# Classification report (f-score, precision, recall, support)
print(classification_report(y_train, y_pred))



# Test Model
# Compute the accuracy of the model when run on the test data
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

## Export model to GraphViz
import graphviz
from graphviz import Graph

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 

graph.render('titanic_decision_tree', view=True)

import eli5
eli5.explain_weights(clf)