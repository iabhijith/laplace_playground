import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from .base import NumpyDataset, between_split


class RegressionDataGenerator:
    def __init__(self, n_samples=1000, n_features=10, noise=5, random_state=4):
        self.random_state = random_state
        self.n_features = n_features
        X, y = datasets.make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state,
        )
        y = y.reshape(-1, 1)
        X_scaler = MinMaxScaler()
        self.X_scaled = X_scaler.fit_transform(X)
        y_scaler = MinMaxScaler()
        self.y_scaled = y_scaler.fit_transform(y)

    def train_test_split(self, test_size=0.33, random_state=None):
        if random_state is None:
            random_state = self.random_state
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_scaled, test_size=test_size, random_state=random_state
        )
        return NumpyDataset(X_train, y_train), NumpyDataset(X_test, y_test)

    def between_split(self, test_size=0.33, on=None, random_state=None):
        if self.n_features > 1 and on is None:
            raise Exception(
                "Require the index of the feature on which the between split is required"
            )
        else:
            X_train, y_train, X_test, y_test = between_split(
                X=self.X_scaled, y=self.y_scaled, test_size=test_size, on=on
            )
            return NumpyDataset(X_train, y_train), NumpyDataset(X_test, y_test)


class SinusoidalDataGenerator:
    def __init__(self, n_samples=1000, sigma_noise=0.02, freq=2.5, random_state=4):
        self.n_samples = n_samples
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        self.freq = freq

    def train_test_split(self, test_size=0.33):
        self.X = np.linspace(-1, 1, self.n_samples).reshape(self.n_samples, 1)
        self.y = np.sin(np.pi * self.freq * self.X) + np.random.normal(
            scale=self.sigma_noise, size=(self.n_samples, 1)
        )
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        return NumpyDataset(X_train, y_train), NumpyDataset(X_test, y_test)

    def between_split(self, test_size=0.33):
        test_sample_size = int(self.n_samples * test_size)
        train_sample_size = self.n_samples - test_sample_size
        cluster_size = int(train_sample_size / 2)
        X_train = np.concatenate(
            [
                np.linspace(-1, -0.75, cluster_size),
                np.linspace(0.65, 1.1, train_sample_size - cluster_size),
            ]
        ).reshape(train_sample_size, 1)
        y_train = np.sin(self.freq * np.pi * (X_train + 0.75)) + np.random.normal(
            scale=self.sigma_noise, size=(train_sample_size, 1)
        )
        X_test = np.linspace(-2, 2, test_sample_size).reshape(test_sample_size, 1)
        y_test = np.sin(self.freq * np.pi * (X_test + 0.75))
        return NumpyDataset(X_train, y_train), NumpyDataset(X_test, y_test)
