### EXTERNALLY PROVIDED ###

# coding: utf-8
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score


def train_and_evaluation(X, y):
    """
    Simple auxiliar method to help validate experiments during pre-processing tasks

    Args:
        X: dataframe with the columns relevants (only numeric columns are accepted)
        y: target variable in [numpy.array or pandas serie] format
        return: model accuracy, value from 0 to 1
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2018)

    model = linear_model.LogisticRegression(solver='liblinear')

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return accuracy_score(y_test, pred)
