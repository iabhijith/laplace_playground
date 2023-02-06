import torch
from .sample_datasets import SinusoidalDataGenerator
from .uci_regression_datasets import UCIDataset
from torch.utils.data import DataLoader, random_split


def get_dataloaders(
    dataset,
    data_path=None,
    split="random",
    on=None,
    seed=99,
    batch_size=64,
    num_workers=2,
    shuffle=True,
    val=False,
):
    if dataset == "sinusoidal":
        generator = SinusoidalDataGenerator(1000, 0.075, freq=1, random_state=seed)
        if split == "random":
            train_dataset, test_dataset = generator.train_test_split(test_size=0.33)
        else:
            train_dataset, test_dataset = generator.between_split(test_size=0.33)
    else:
        data = UCIDataset(name=dataset, data_path=data_path)
        train_dataset, test_dataset = data.get_datasets(
            test_size=0.33, random_state=seed, split="between", on=on
        )
        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
        )
    if val:
        train_dataset, val_dataset = split_val(train_dataset, seed=seed)
        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
        )

    else:
        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
        )


def split_val(dataset, val_ratio=0.2, seed=99):
    test_ratio = 1 - val_ratio
    train_dataset_size = int(len(train_dataset) * test_ratio)
    valid_dataset_size = len(dataset) - train_dataset_size
    gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_dataset_size, valid_dataset_size], generator=gen
    )
