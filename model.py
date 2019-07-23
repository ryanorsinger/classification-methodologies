# Logistic Regression
#     Fit the logistic regression classifier to your training sample and transform, i.e. make predictions on the training sample
#     Evaluate your in-sample results using the model score, confusion matrix, and classification report.
#     Print and clearly label the following: Accuracy, true positive rate, false positive rate, true negative rate, false negative rate, precision, recall, f1-score, and support.
#     Look in the scikit-learn documentation to research the solver parameter. What is your best option(s) for the particular problem you are trying to solve and the data to be used?
#     Run through steps 2-4 using another solver (from question 5)
#     Which performs better on your in-sample data?
#     Save the best model in logit_fit

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = data('iris')
df.head()

df.columns = [col.lower().replace('.', '_') for col in df]

X = df.drop(['species'],axis=1)
y = df[['species']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)

X_train.head()

# for classificaiton you can change the algorithm as gini or entropy (information gain).  Default is gini.
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)

# Fit the model to the training data
clf.fit(X_train, y_train)

DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=123,
            splitter='best')

y_pred = clf.predict(X_train)
y_pred[0:5]

y_pred_proba = clf.predict_proba(X_train)
y_pred_proba

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))

confusion_matrix(y_train, y_pred)

sorted(y_train.species.unique())

y_train.species.value_counts()

labels = sorted(y_train.species.unique())

pd.DataFrame(confusion_matrix(y_train, y_pred), index=labels, columns=labels)

print(classification_report(y_train, y_pred))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


from sklearn.datasets import load_iris

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

import graphviz

from graphviz import Graph

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 

graph.render('iris_decision_tree', view=True)