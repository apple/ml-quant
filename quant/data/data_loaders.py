#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Data loaders for MNIST, CIFAR-10, CIFAR-100, and ImageNet datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
import typing as t

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


class QuantDataLoader(ABC):
    """Abstract class from which to instantiate training and test set PyTorch data loaders."""

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        download: bool = True,
        test_sampler: t.Optional[Sampler] = None,
    ):
        """
        Construct QuantDataLoader object, used for obtaining training and test set loaders.

        Args:
            train_batch_size: training set batch size
            test_batch_size: test set batch size
            dataset_path: root location of the dataset
            workers: number of workers to use for the data loader
            download: whether to download dataset.
                If false `dataset_path` should contain pre-downloaded dataset.
            test_sampler: PyTorch data sampler for the test set
        """
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dataset_path = dataset_path
        self.workers = workers
        self.download = download
        self.test_sampler = test_sampler

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        raise NotImplementedError

    @abstractmethod
    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Clean up any temporary data."""
        pass


class MNISTDataLoader(QuantDataLoader):
    """
    Subclass of :class:`~quant.data.data_loaders.QuantDataLoader`, for MNIST.

    If the `dataset_path` does not already have the dataset, it is downloaded from the web.
    """

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        download: bool = True,
        test_sampler: t.Optional[Sampler] = None,
    ):
        """Construct a class for getting MNIST data loaders."""
        super(MNISTDataLoader, self).__init__(
            train_batch_size,
            test_batch_size,
            dataset_path,
            workers,
            download,
            test_sampler,
        )
        self.transform_fn = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.dataset_path,
                train=True,
                download=self.download,
                transform=self.transform_fn,
            ),
            batch_size=self.train_batch_size,
            shuffle=True,
        )

        return train_loader

    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.dataset_path,
                train=False,
                download=self.download,
                transform=self.transform_fn,
            ),
            batch_size=self.test_batch_size,
            shuffle=False,
            sampler=self.test_sampler,
        )

        return test_loader


class CIFAR10DataLoader(QuantDataLoader):
    """
    Subclass of :class:`~quant.data.data_loaders.QuantDataLoader`, for CIFAR-10.

    If the `dataset_path` does not already have the dataset, it is downloaded from the web.
    """

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        download: bool = True,
        test_sampler: t.Optional[Sampler] = None,
    ):
        """Construct a class for getting CIFAR-10 data loaders."""
        super(CIFAR10DataLoader, self).__init__(
            train_batch_size,
            test_batch_size,
            dataset_path,
            workers,
            download,
            test_sampler,
        )
        self.mean_val = (0.4914, 0.4822, 0.4465)
        self.std_val = (0.2023, 0.1994, 0.2010)

    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean_val, self.std_val),
            ]
        )

        dataset_train = datasets.CIFAR10(
            root=self.dataset_path,
            train=True,
            download=self.download,
            transform=transform_train,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

        return train_loader

    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean_val, self.std_val)]
        )

        dataset_test = datasets.CIFAR10(
            root=self.dataset_path,
            train=False,
            download=self.download,
            transform=transform_test,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=self.workers,
            pin_memory=True,
        )

        return test_loader


class CIFAR100DataLoader(QuantDataLoader):
    """
    Subclass of :class:`~quant.data.data_loaders.QuantDataLoader`, for CIFAR-100.

    If the `dataset_path` does not already have the dataset, it is downloaded from the web.
    """

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        download: bool = True,
        test_sampler: t.Optional[Sampler] = None,
    ):
        """Construct a class for getting CIFAR-100 data loaders."""
        super(CIFAR100DataLoader, self).__init__(
            train_batch_size,
            test_batch_size,
            dataset_path,
            workers,
            download,
            test_sampler,
        )
        self.mean_val = (0.507075159237, 0.4865488733149, 0.440917843367)
        self.std_val = (0.267334285879, 0.2564384629170, 0.276150471325)

    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean_val, self.std_val),
            ]
        )

        dataset_train = datasets.CIFAR100(
            root=self.dataset_path,
            train=True,
            download=self.download,
            transform=transform_train,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

        return train_loader

    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean_val, self.std_val)]
        )

        dataset_test = datasets.CIFAR100(
            root=self.dataset_path,
            train=False,
            download=self.download,
            transform=transform_test,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=self.workers,
            pin_memory=True,
        )

        return test_loader


class ImageNetDataLoader(QuantDataLoader):
    """
    Subclass of :class:`~quant.data.data_loaders.QuantDataLoader`, for ImageNet.

    The dataset must already be available and cannot be downloaded by this data loader.
    """

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        download: bool = False,
        test_sampler: t.Optional[Sampler] = None,
        train_split: str = 'train',
        val_split: str = 'val',
    ):
        """Construct a class for getting ImageNet data loaders."""
        super(ImageNetDataLoader, self).__init__(
            train_batch_size,
            test_batch_size,
            dataset_path,
            workers,
            download,
            test_sampler,
        )
        if download:
            raise ValueError(
                'ImageNet must be downloaded manually due to licensing restrictions.'
            )
        self.train_split = train_split
        self.val_split = val_split

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        train_dir = Path(self.dataset_path) / self.train_split
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            ),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

        return train_loader

    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""
        test_dir = Path(self.dataset_path) / self.val_split
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            ),
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            sampler=self.test_sampler,
        )

        return test_loader
