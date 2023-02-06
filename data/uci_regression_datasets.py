import numpy as np
import torch
import torchvision.datasets as datasets
import os
from os import path
import pandas as pd
import zipfile
import urllib.request

from sklearn.model_selection import train_test_split
from .base import NumpyDataset, between_split


DATASETS = {
    "housing": {
        "uri": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        "type": "file",
        "file_type": "csv",
        "header": False,
        "delimiter": "\s+",
    },
    "power": {
        "uri": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        "type": "zip",
        "file_name": "Folds5x2_pp.xlsx",
        "file_type": "excel",
        "header": False,
    },
    "wine": {
        "uri": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "type": "file",
        "file_type": "csv",
        "header": True,
        "delimiter": ";",
    },
    "yacht": {
        "uri": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        "type": "file",
        "file_type": "csv",
        "header": False,
        "delimiter": "\s+",
    },
    "naval": {
        "uri": "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
        "type": "zip",
        "file_name": "data.txt",
        "file_type": "csv",
        "header": False,
        "delimiter": "\s+",
    },
    "energy": {
        "uri": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "type": "file",
        "file_type": "excel",
        "header": False,
        "delimiter": "\s+",
    },
}


class UCIDataset:
    def __init__(self, name, data_path=None):
        self.name = name
        if self.name not in DATASETS:
            raise Exception("Dataset URI not available")

        self.data_path = data_path
        if self.data_path is None:
            self.data_path = "UCI"
        os.mkdirs(path=self.data_path, exist_ok=True)

        self.data = self.read_data()

    def read_data(self):
        dataset = DATASETS[self.name]
        uri = dataset["uri"]
        file_name = uri.split("/")[-1]
        file_path = os.path.join(self.data_path, file_name)
        if not path.exists(file_path):
            urllib.request.urlretrieve(DATASETS[self.name], file_path)

        if dataset["type"] == "zip":
            unzip_to = os.path.join(self.data_path, file_name.split(".")[0])
            zipfile.ZipFile(file_path).extractall(unzip_to)
            file_path = os.path.join(unzip_to, dataset["file_name"])

        if dataset["file_type"] == "csv":
            data = pd.read_csv(
                file_path, header=dataset["header"], delimiter=dataset["delimiter"]
            ).values
        elif dataset["file_type"] == "excel":
            data = pd.read_excel(file_path, header=dataset["header"]).values

        return data

    def get_datasets(self, test_size=0.33, random_state=42, split="random", on=None):
        if split == "random":
            if random_state is None:
                random_state = self.random_state
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=random_state
                )
        return NumpyDataset(X_train, y_train), NumpyDataset(X_test, y_test)
