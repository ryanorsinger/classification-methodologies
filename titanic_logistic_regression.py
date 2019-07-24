
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from acquire import get_titanic_data
from prepare import prepare_titanic_data

df = prepare_titanic_data(get_titanic_data())

X = df[['pclass','age','fare','sibsp','parch']]
y = df[['survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)

X_train.head()

# Create the logistic regression object
logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')

# Fit the model to the training data
logit.fit(X_train, y_train)

print('Coefficient: \n', logit.coef_)
print('Intercept: \n', logit.intercept_)

# Pull in the eli5 model explanation tool
import eli5
eli5.show_weights(logit)

# Estimate whether or not a passenger would survive, using the training data
y_pred = logit.predict(X_train)

# Estimate the probability of a passenger surviving, using the training data
y_pred_proba = logit.predict_proba(X_train)

print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logit.score(X_train, y_train)))

# Create a confusion matrix
print(confusion_matrix(y_train, y_pred))

# Compute Precision, Recall, F1-score, and Support
print(classification_report(y_train, y_pred))

print('Accuracy of Logistic Regression classifier on test set: {:.2f}'
     .format(logit.score(X_test, y_test)))

## verify
y_pred_proba = [i[1] for i in y_pred_proba]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_pred_proba, y_pred)
