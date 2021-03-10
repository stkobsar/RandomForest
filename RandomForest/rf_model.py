from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np


def data_train_test_split(df):
    X = df.iloc[:, 0:29].values
    y = df.iloc[:, 30].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test


def data_scaling(X_train, X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    return X_train_scaled, X_test_scaled

def data_preparing(df):

    X_train, X_test, y_train, y_test = data_train_test_split(df)
    X_train_scaled, X_test_scaled = data_scaling(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def RandomForestModel(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    crossval_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=10))
    model_accuracy = metrics.accuracy_score(y_test, y_pred)

    return confusion_matrix, crossval_score, model_accuracy
