#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test parser."""

import pytest
import torch

from quant.common.parser import get_base_argument_parser, parse_config


@pytest.fixture()
def base_parser():
    """Fixture for base argument parser."""
    return get_base_argument_parser('base parser')


def test_standard_args(base_parser):
    """Test parsing standard arguments."""
    args = base_parser.parse_args('--config examples/mnist/mnist_fp.yaml'.split(' '))
    config = parse_config(args)

    assert isinstance(config['experiment_name'], str) and len(config['experiment_name'])
    assert config['environment']['platform'] == 'local'
    assert config['environment']['ngpus'] == (1 if torch.cuda.is_available() else 0)
    assert 'init_from_checkpoint' not in config
    assert 'restore_experiment' not in config
    assert not config['skip_training']


def test_missing_config(base_parser):
    """Test missing config."""
    args = base_parser.parse_args([])
    with pytest.raises(ValueError):
        parse_config(args)


def test_gpu_override(base_parser):
    """Test CLI ngpus argument can override what is in the config."""
    args = base_parser.parse_args('--config examples/mnist/mnist_fp.yaml --ngpus 8'.split(' '))
    config = parse_config(args)

    assert config['environment']['ngpus'] == 8
