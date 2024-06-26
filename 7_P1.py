import numpy as np
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

    np.random.seed(random_state)

    idx_obj = np.random.choice(range(len(X[:, 0])), replace=True, size=len(X[:, 0]))
    idx_obj = np.unique(idx_obj)

    return idx_obj.astype(int)
    pass
  @staticmethod
  def random_subspace(X, n_subspace, random_state=42):

    np.random.seed(random_state)

    idx_subspace = np.random.choice(range(len(X[0, :])), replace=False, size=n_subspace)
    return idx_subspace.astype(int)

    pass

  @staticmethod
  def get_subsample(X, y, idx_subspace, idx_obj):

    x_sampled = np.empty([len(idx_obj), len(idx_subspace)])
    y_sampled = np.empty([len(idx_obj)])

    for i in range(len(idx_obj)):
      for j in range(len(idx_subspace)):
        x_sampled[i, j] = X[idx_obj[i], idx_subspace[j]]
        y_sampled[i] = y[idx_obj[i]]

    return x_sampled, y_sampled

    pass




X = np.array([[1,2,3], [4,5,6], [7,8,9]])
Y = np.array([1, 2, 3])
s = sample(X, 2)

bootstrap_indices = s.bootstrap_sample(X)
X_sampled, y_sampled = s.get_subsample(X, Y, s.idx_subspace, bootstrap_indices)

print(bootstrap_indices)

print(s.idx_subspace)

print(X_sampled)
print(y_sampled)


