#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Tests for data loaders."""

import pytest
from torch.utils.data.sampler import SubsetRandomSampler

from quant.data.data_loaders import (
    MNISTDataLoader,
    CIFAR10DataLoader,
    CIFAR100DataLoader,
)


@pytest.mark.slow
def test_mnist_data_loader(tmp_path_factory):
    """Test MNIST data loading."""
    mnist_dir = tmp_path_factory.getbasetemp() / 'MNIST'
    for download in (True, False):
        mnist_data_loader = MNISTDataLoader(32, 32, mnist_dir, 4, download=download)

        train_loader = mnist_data_loader.get_train_loader()
        assert len(train_loader.dataset) == 60000

        test_loader = mnist_data_loader.get_test_loader()
        assert len(test_loader.dataset) == 10000

    subset_up_to = 64
    sampler = SubsetRandomSampler(range(subset_up_to))
    mnist_data_loader = MNISTDataLoader(
        32, 32, mnist_dir, 4, download=False, test_sampler=sampler
    )

    test_loader = mnist_data_loader.get_test_loader()
    assert len(test_loader) == subset_up_to / 32


@pytest.mark.slow
def test_cifar10_data_loader(tmp_path_factory):
    """Test CIFAR-10 data loading."""
    cifar10_dir = tmp_path_factory.getbasetemp() / 'CIFAR-10'
    for download in (True, False):
        cifar10_data_loader = CIFAR10DataLoader(
            32, 32, cifar10_dir, 4, download=download
        )

        train_loader = cifar10_data_loader.get_train_loader()
        assert len(train_loader.dataset) == 50000

        test_loader = cifar10_data_loader.get_test_loader()
        assert len(test_loader.dataset) == 10000

    subset_up_to = 64
    sampler = SubsetRandomSampler(range(subset_up_to))
    cifar10_data_loader = CIFAR10DataLoader(
        32, 32, cifar10_dir, 4, download=False, test_sampler=sampler
    )

    test_loader = cifar10_data_loader.get_test_loader()
    assert len(test_loader) == subset_up_to / 32


@pytest.mark.slow
def test_cifar100_data_loader(tmp_path_factory):
    """Test CIFAR-100 data loading."""
    cifar100_dir = tmp_path_factory.getbasetemp() / 'CIFAR-100'
    for download in (True, False):
        cifar100_data_loader = CIFAR100DataLoader(
            32, 32, cifar100_dir, 4, download=download
        )

        train_loader = cifar100_data_loader.get_train_loader()
        assert len(train_loader.dataset) == 50000

        test_loader = cifar100_data_loader.get_test_loader()
        assert len(test_loader.dataset) == 10000

    subset_up_to = 64
    sampler = SubsetRandomSampler(range(subset_up_to))
    cifar100_data_loader = CIFAR100DataLoader(
        32, 32, cifar100_dir, 4, download=False, test_sampler=sampler
    )

    test_loader = cifar100_data_loader.get_test_loader()
    assert len(test_loader) == subset_up_to / 32
