#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test for running tasks."""

from quant.common.tasks import classification_task
from quant.utils.visualization import get_tensorboard_hooks
import pytest
import yaml

from tests.data.helpers import get_base_config_template, RandomQuantDataLoader


@pytest.mark.incremental
class TestRunClassificationTask:

    arch_variants = [
        {'x_quant': 'fp', 'w_quant': 'fp'},
        {'x_quant': 'ls-2', 'w_quant': 'ls-1'},
        {'x_quant': 'gf-2', 'w_quant': 'ls-1'},
        {'x_quant': 'ls-2', 'w_quant': 'ls-1',
         'moving_average_mode': 'eval_only', 'moving_average_momentum': 0.9},
        {'x_quant': 'ls-1', 'w_quant': 'ls-1',
         'moving_average_mode': 'train_and_eval', 'moving_average_momentum': 0.9}
    ]

    def test_train_regular_classification_task(self, tmp_path_factory):
        """Train a model from scratch, which will be used as the teacher."""
        for i, arch_variant in enumerate(self.arch_variants):
            base_dir = tmp_path_factory.getbasetemp()
            config = get_base_config_template(base_dir, f'teacher_{i}', arch_variant)
            classification_task(
                config,
                base_dir / 'experiments',
                RandomQuantDataLoader,
                get_tensorboard_hooks
            )

            with open(str(base_dir / 'experiments' / f'teacher_{i}' / 'config.yaml'), 'w') as f:
                yaml.dump(config, f)

    def test_init_from_checkpoint(self, tmp_path_factory):
        """Test initializing from checkpoint."""
        for i, arch_variant in enumerate(self.arch_variants):
            base_dir = tmp_path_factory.getbasetemp()
            config = get_base_config_template(base_dir, f'init_from_checkpoint_{i}', arch_variant)
            config['init_from_checkpoint'] = str(
                base_dir / 'experiments' / f'teacher_{i}' / 'checkpoints' / 'checkpoint_1.pt'
            )
            classification_task(
                config,
                base_dir / 'experiments',
                RandomQuantDataLoader,
                get_tensorboard_hooks
            )

    def test_skip_training(self, tmp_path_factory):
        """Test only doing inference."""
        for i, arch_variant in enumerate(self.arch_variants):
            base_dir = tmp_path_factory.getbasetemp()
            config = get_base_config_template(base_dir, f'skip_training_{i}', arch_variant)
            config['skip_training'] = True
            config['init_from_checkpoint'] = str(
                base_dir / 'experiments' / f'teacher_{i}' / 'checkpoints' / 'checkpoint_1.pt'
            )
            classification_task(
                config,
                base_dir / 'experiments',
                RandomQuantDataLoader,
                get_tensorboard_hooks
            )

    def test_restore_from_experiment(self, tmp_path_factory):
        """Test restoring from experiment."""
        for i, arch_variant in enumerate(self.arch_variants):
            base_dir = tmp_path_factory.getbasetemp()
            config = get_base_config_template(base_dir, f'restore_experiment_{i}', arch_variant)
            classification_task(
                config,
                base_dir / 'experiments',
                RandomQuantDataLoader,
                get_tensorboard_hooks,
                base_dir / 'experiments' / f'teacher_{i}'
            )

    def test_train_student(self, tmp_path_factory):
        """Train a student model using the teacher from above."""
        for i, arch_variant in enumerate(self.arch_variants):
            base_dir = tmp_path_factory.getbasetemp()
            config = get_base_config_template(base_dir, f'student_{i}', arch_variant)
            config['model']['kd_config'] = {
                'teacher_config_path': str(
                    base_dir / 'experiments' / f'teacher_{i}' / 'config.yaml'
                ),
                'teacher_checkpoint_path': str(
                    base_dir / 'experiments' / f'teacher_{i}' / 'checkpoints' / 'checkpoint_1.pt'
                ),
                'freeze_teacher': True,
                'train_mode': True,
                'criterion_config': {'temperature': 1}
            }

            classification_task(
                config,
                base_dir / 'experiments',
                RandomQuantDataLoader,
                get_tensorboard_hooks
            )
