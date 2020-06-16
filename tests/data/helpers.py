#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Helpers for data loader tests."""

import typing as t

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader

from quant.data.data_loaders import QuantDataLoader


class RandomDataset(Dataset):
    def __init__(self, num_classes: int):
        self.nc = num_classes

    def __len__(self):
        return 256

    def __getitem__(self, index):
        # return (data, target) as a tuple
        return torch.normal(mean=0, std=1, size=(1, 28, 28)), torch.randint(0, self.nc, (1,)).item()


class RandomQuantDataLoader(QuantDataLoader):

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        download: bool = False,
        test_sampler: t.Optional[Sampler] = None,
        num_classes: int = 10,
    ):
        """Construct a class for getting RandomQuantDataLoader data loaders."""
        super(RandomQuantDataLoader, self).__init__(
            train_batch_size,
            test_batch_size,
            dataset_path,
            workers,
            download,
            test_sampler,
        )
        self.num_classes = num_classes

    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        train_loader = DataLoader(
            RandomDataset(self.num_classes), batch_size=self.train_batch_size, shuffle=False
        )

        return train_loader

    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""
        test_loader = DataLoader(
            RandomDataset(self.num_classes), batch_size=self.test_batch_size,
            shuffle=False, sampler=self.test_sampler
        )

        return test_loader


def get_base_config_template(tmp_path, exp_name, arch_variant):
    base_template = {
        'environment': {'ngpus': 0},
        'experiment_name': exp_name,
        'skip_training': False,
        'data': {
            'dataset_path': str(tmp_path / 'data'),
            'train_batch_size': 64,
            'test_batch_size': 64,
            'workers': 4
        },
        'model': {
            'architecture': 'lenet5',
            'loss': 'nll_loss',
            'arch_config': {
                'conv1_filters': 2,
                'conv2_filters': 5,
                'output_classes': 10
            }
        },
        'optimization': {
            'epochs': 1,
            'optimizer': {
                'algorithm': 'adadelta',
                'lr': 0.1
            },
            'lr_scheduler': {
                'scheduler': 'step_lr',
                'step_size': 1,
                'gamma': 0.9
            }
        },
        'log': {
            'level': 'INFO',
            'interval': 10,
            'tensorboard': True,
            'tensorboard_root': str(tmp_path / 'tb_runs'),
            'root_experiments_dir': str(tmp_path / 'experiments'),
            'save_model_freq': 1
        }
    }

    base_template['model']['arch_config'].update(arch_variant)

    return base_template
