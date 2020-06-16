#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
Configurations for Quant.

All configurations are specified in YAML files.
Certain parameters such as the number of GPUs can be overridden or specified from the CLI.

The config is divided into sections::

    seed: (int or null)
    environment: ...
    data: ...
    model: ...
    optimization: ...
    log: ...

See `examples/mnist/mnist.yaml` for an example.

The model architecture, loss criterion, optimizer, and learning rate scheduler are all specified
in the YAML config.

Environment
^^^^^^^^^^^

This section specifies the computing environment and resources.

`platform` should always be set to `local`.
One can subclass :class:`~quant.common.compute_platform.ComputePlatform`
and create alternate platforms to train models on (such as some cloud GPU server).
If you do this `platform` can be set to something else to distinguish it from `local`.

`ngpus` specified the number of GPUs to use.

The `cuda` subsection can be configured to set CUDA configurations, for example::

    cuda:
        cudnn_deterministic: false
        cudnn_benchmark: true

Data
^^^^

The data section sets the dataset location, batch sizes, and number of workers dataset loading.

Here is an example::

    data:
        dataset_path: data/imagenet/
        train_batch_size: 256
        test_batch_size: 256
        workers: 16

Model
^^^^^

This section specifies the model architecture and loss::

    model:
        architecture: lenet5
        loss: nll_loss
        arch_config: ...

Supported architectures include: `lenet5` and `resnet`.
Supported loss functions include: `cross_entropy`, `nll_loss`, `kl_div`.
Architecture config stores keyword arguments passed to the model constructor.
See model constructor documentation (:class:`~quant.models.lenet.QLeNet5` or
:class:`~quant.models.resnet.QResNet`) for more info.

For training with teacher, one can add another subsection under `model`, such as::

    kd_config:
        teacher_config_path: examples/imagenet/imagenet_fp.yaml
        teacher_checkpoint_path: experiments/imagenet-teacher/checkpoints/checkpoint_100.pt
        freeze_teacher: true
        train_mode: true
        criterion_config:
            temperature: 1

Optimization
^^^^^^^^^^^^

This section specifies configurations for the optimizer and learning rate scheduler, for example::

    optimization:
        epochs: 14
        optimizer:
            algorithm: adadelta
            lr: 1.0
        lr_scheduler:
            scheduler: step_lr
            step_size: 1
            gamma: 0.7

Optimization algorithms (`algorithm`) support include: `sgd`, `adam`, `adadelta`.
All other key-value pairs under `optimizer` are passed directly as keyword arguments to
the corresponding PyTorch optimizer class's constructor:
https://pytorch.org/docs/stable/optim.html#algorithms.

Learning rate scheduler (`scheduler`) support include:
`linear_lr`, `lambda_lr`, `step_lr`, and `multi_step_lr`.
All other key-value pairs under `lr_scheduler` are passed directly as keyword arguments to
the corresponding PyTorch LR scheduler class's constructor:
https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate.
See more details about the scheduler configurations at
:meth:`~quant.common.initialization.get_lr_scheduler`.

Log
^^^

This section specifies the configurations for logging, checkpointing, and visualization.

A sample config looks like this::

    log:
        level: INFO
        interval: 100
        tensorboard: true
        tensorboard_root: runs/
        root_experiments_dir: experiments/
        save_model_freq: 20

`interval` is number of batches per print of the current metrics to STDOUT and TensorBoard.

If `tensorboard` is true, TensorBoard will be used to visualize metrics.
`tensorboard_root` is the location of all TensorBoard logs. The location for
visualization logs of one experiment will be under a subdirectory with the experiment name.

`root_experiments_dir` is the root location for storing all experiment logs.
The logs for one experiment will be stored under a subdirectory with the experiment name,
which can be passed in via the CLI (or omit to use default).

The experiment directory will contain the resolved config, overall metrics, checkpoints,
and copy of TensorBoard logs.

`save_model_freq` is the number of epochs between saving checkpoints.
The last epoch is always saved.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable
import yaml

import torch


def _validate_args(args: Namespace) -> None:
    """
    Validate arguments.

    Args:
        args:  parsed argparse CLI args
    """
    if not args.restore_experiment and not args.config:
        raise ValueError('--config must be specified if not restoring from experiment.')

    if args.restore_experiment and args.init_from_checkpoint:
        raise ValueError('Only one of --restore-experiment / --init-from-checkpoint can be set.')


def parse_common_fields(args: Namespace, config: dict) -> None:
    """
    Populate common fields in the config with parsed args.

    Args:
        args: parsed argparse CLI args
        config: config dictionary storing final resolved args
    """
    if args.experiment_name is not None:
        config['experiment_name'] = args.experiment_name
    else:
        from datetime import datetime

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        config_name_without_ext = Path(config['config']).stem
        config['experiment_name'] = f'{current_time}_{config_name_without_ext}'

    if 'environment' not in config or 'platform' not in config['environment']:
        config['environment'] = {'platform': 'local'}

    if args.ngpus is not None:
        config['environment']['ngpus'] = args.ngpus
    if 'ngpus' not in config['environment']:
        config['environment']['ngpus'] = 1 if torch.cuda.is_available() else 0

    config['skip_training'] = args.skip_training

    if args.init_from_checkpoint:
        config['init_from_checkpoint'] = args.init_from_checkpoint


def parse_config(args: Namespace, validator: Callable[[Namespace], None] = _validate_args) -> dict:
    """
    Parse config file and override with CLI args.

    Args:
        args: parsed argparse CLI args
        validator: validator for config

    Returns:
        A resolved config, applying CLI args on top of the config file
    """
    validator(args)

    config = {}
    if args.restore_experiment:
        with open(Path(args.restore_experiment) / 'config.yaml') as f:
            config = yaml.safe_load(f)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config['config'] = args.config

    parse_common_fields(args, config)

    if args.restore_experiment:
        config['restore_experiment'] = args.restore_experiment

    return config


def get_base_argument_parser(description: str) -> ArgumentParser:
    """
    Get a base argument parser for driver scripts.

    Args:
        description: A string describing the driver script.

    Returns:
        Parser object to extend.
    """
    parser = ArgumentParser(description)
    parser.add_argument('--config', type=str, help='Path to a yaml config file.')
    parser.add_argument(
        '--experiment-name', type=str, default=None, help='Name of the experiment.'
    )
    parser.add_argument(
        '--ngpus', type=int, default=None, help='Number of GPUs. Use 0 for CPU.'
    )
    parser.add_argument(
        '--skip-training',
        default=False,
        action='store_true',
        help='Skip training and only run evaluation. Checkpoint must be passed in as well.',
    )
    parser.add_argument(
        '--restore-experiment',
        type=str,
        help='Path to experiments directory to restore checkpoint from.',
    )
    parser.add_argument(
        '--init-from-checkpoint',
        type=str,
        help='Path to model file to initialize model parameters.',
    )
    return parser
