.. currentmodule:: quant.common

Common
======

This module contains common code for running the code, performing training and evaluation.

.. note::

    If you are just running the example code to reproduce the paper, you do not need to read
    the sections below :ref:`Config File` and :ref:`CLI Args`. If you want to write your own
    driver scripts that use Quant for your tasks, you may find the additional documentation
    helpful.

.. _Config File:

Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quant.common.parser

.. _CLI Args:

CLI Args
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can always use ``--help`` if running any of the example scripts to see the arguments.

``--config <path to YAML config file>`` specifies the path to the yaml config file.

The experiment can be given a name with ``--experiment-name <name>``.
If no name is specified a name is chosen based on the dataset name and time.

``--ngpus <number of GPUs>`` can be used to set or override the number of GPUs setting
in the config.

``--init-from-checkpoint <path to .pt>`` can be used to initialize the model from a checkpoint.
See :meth:`~quant.utils.checkpoints.restore_from_checkpoint` for more details.
This only stores the model from the checkpoint, but not the optimizer or scheduler state.

Alternatively, ``--restore-checkpoint <path to experiment directory>`` can be used
to resume training from a checkpoint. The last checkpoint will be used.

If either ``--init-from-checkpoint`` or ``--restore-checkpoint`` is used,
``--skip-training`` can be set to perform only inference on the test set.

Initializing Device, Model, and Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quant.common.initialization
    :members:
    :special-members: __init__
    :undoc-members:

Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quant.common.experiment
    :members:
    :special-members: __init__
    :undoc-members:

Compute Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quant.common.compute_platform
    :members:
    :special-members: __init__
    :undoc-members:

Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quant.common.metrics
    :members:
    :special-members: __init__
    :undoc-members:

Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quant.common.training
    :members:
