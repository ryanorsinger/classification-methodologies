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

from acquire import get_iris_data
from prepare import prepare_iris_data

df = prepare_iris_data(get_iris_data())

X = df.drop(columns=["species"])
y = df[['species']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)

X_train.head()

# Create the Logistic Regression Object
logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')

# Fit the model to the training data
logit.fit(X_train, y_train)

# Print the coefficients and intercept of the model
print('Coefficient: \n', logit.coef_)
print('Intercept: \n', logit.intercept_)

# Change the comment to reflect iris data
# Estimate whether or not a passenger would survive, using the training data

y_pred = logit.predict(X_train)

# Change the comment to reflect iris data
# Estimate the probability of a passenger surviving, using the training data

y_pred_proba = logit.predict_proba(X_train)

# Evaluate Model
# Compute the accuracy

print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logit.score(X_train, y_train)))

# Create a confusion matrix

print(confusion_matrix(y_train, y_pred))

# Compute Precision, Recall, F1-score, and Support
print(classification_report(y_train, y_pred))

# Compute the accuracy of the model when run on the test data
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(logit.score(X_test, y_test)))

## verify
y_pred_proba = [i[1] for i in y_pred_proba]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_pred_proba, y_pred)
