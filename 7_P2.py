
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random

np.random.seed(42)

N_ESTIMATORS = 10
MAX_DEPTH = 5
SUBSPACE_DIM = 2

class random_forest:
    def __init__(self, n_estimators=10, max_depth=None, subspaces_dim=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state
        self._estimators = []
        self.subspace_idx = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            n_features = X.shape[1]
            sample_indices = np.random.choice(range(len(X)), size=len(X))
            bootstrap_X = X[sample_indices]
            bootstrap_y = y[sample_indices]


            if self.subspaces_dim and self.subspaces_dim <= n_features:
                features_idx = np.random.choice(range(n_features), size=self.subspaces_dim, replace=False)
            else:
                features_idx = range(n_features)

            self.subspace_idx.append(features_idx)


            clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state + i)
            clf.fit(bootstrap_X[:, features_idx], bootstrap_y)
            self._estimators.append(clf)

    def predict(self, X):

        predictions = np.zeros((X.shape[0], len(self._estimators)))

        for i, clf in enumerate(self._estimators):
            features_idx = self.subspace_idx[i]
            predictions[:, i] = clf.predict(X[:, features_idx])


        final_predictions = [np.argmax(np.bincount(predictions[i].astype(int))) for i in range(len(predictions))]
        return final_predictions











'''import numpy as np
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)


class sample(object):
    def __init__(self, X, n_subspace):
        self.idx_subspace = self.random_subspace(X, n_subspace)

    def __call__(self, X, y):
        idx_obj = self.bootstrap_sample(X)
        X_sampled, y_sampled = self.get_subsample(X, y, self.idx_subspace, idx_obj)
        return X_sampled, y_sampled

    @staticmethod
    def bootstrap_sample(X, random_state=42):

        idx_obj = np.random.choice(range(len(X[:, 0])), replace=True, size=len(X[:, 0]))
        idx_obj = np.unique(idx_obj)

        return idx_obj.astype(int)
        pass


    def random_subspace(X, n_subspace, random_state=42):

        idx_subspace = np.random.choice(range(len(X[0, :])), replace=False, size=n_subspace)
        return idx_subspace.astype(int)

        pass


    def get_subsample(X, y, idx_subspace, idx_obj):

        x_sampled = np.empty([len(idx_obj), len(idx_subspace)])
        y_sampled = np.empty([len(idx_obj)])

        for i in range(len(idx_obj)):
            for j in range(len(idx_subspace)):
                x_sampled[i, j] = X[idx_obj[i], idx_subspace[j]]
                y_sampled[i] = y[idx_obj[i]]

        return x_sampled, y_sampled

        pass


N_ESTIMATORS = None
MAX_DEPTH = None
SUBSPACE_DIM = None


class random_forest(object):
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state

        self._estimators = []
        for i in range(n_estimators):
            self._estimators.append(DecisionTreeClassifier(max_depth=max_depth))
        """
      Задайте все необходимые поля в рамках конструктора класса
    """

    def fit(self, X, y):
        for i in range(self.n_estimators):
            s = sample(X, self.subspaces_dim)
            bootstrap_indices = s.bootstrap_sample(X)
            X_sampled, y_sampled = s.get_subsample(X, y, s.idx_subspace, bootstrap_indices)
            self._estimators[i].fit(X_sampled, y_sampled)

            """
        Напишите функцию обучения каждого из деревьев алгоритма на собственной подвыборке
      """

            pass

    def predict(self, X):
        sumOfPreds = 0
        for i in range(self.n_estimators):
            sumOfPreds += self._estimators[i].predict(X)

        return sumOfPreds / self.n_estimators

        pass
'''