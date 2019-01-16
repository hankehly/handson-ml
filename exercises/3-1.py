import os

import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    knn_clf = KNeighborsClassifier()

    param_grid = [{'weights': ['uniform', 'distance'], 'n_neighbors': [2, 3, 4, 5]}]

    grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, cv=5, verbose=5)
    grid_search.fit(X_train, y_train)

    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '3-1.joblib')
    joblib.dump(grid_search, output_path)
    print('Result stored in {out}'.format(out=output_path))
