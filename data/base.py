import torch
import numpy as np

from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.len

    def get_arrays(self):
        return self.X.squeeze().numpy(), self.y.squeeze().numpy()


def between_split(X, y, test_size, on):
    n, d = X.shape
    sort_feature = X[:, on]
    sorted_indices = np.argsort(sort_feature)
    X = X[sorted_indices]
    y = y[sorted_indices]
    start = int(n * test_size)
    X_train = np.concatenate((X[:start], X[2 * start :]), axis=0)
    X_test = X[start : 2 * start]
    y_train = np.concatenate((y[0:start], y[2 * start :]), axis=0)
    y_test = y[start : 2 * start]
    return X_train, y_train, X_test, y_test
