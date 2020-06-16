#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""
This module contains classes for logging evaluation metrics.

Metric is a stateful object that allows computing collective statistics
on an arbitrary subset of the dataset by implementing three methods:
:meth:`Metric.update`, :meth:`Metric.compute`, and :meth:`Metric.reset`.

For example, :class:`TopKAccuracy` is an evaluation metric for computing the top-`k` accuracy.
All metrics should subclass the :class:`Metric` abstract class.
Each metric has three key methods for updating and getting its value:
:meth:`Metric.update`, :meth:`Metric.compute`, and :meth:`Metric.reset`.

In each epoch, we generally need to iterate the dataset in batches.
Based on the predictions of examples in each batch, we need to update the value of
metrics accordingly.
The :meth:`Metric.update` method is used to update the metric
based on the results of new observations in each batch.
Depending on whether we set `accumulate` to `True` when creating the metric,
the metric is either accumulated with the result of the current batch or overwritten with it.

Once we finish collecting the predictions, it is time to compute the value of the metric
of the whole dataset using :meth:`Metric.compute`.
Before moving on to the next epoch, we may want to :meth:`Metric.reset` the metric so
that we start evaluating on the predictions of the new epoch afresh.

A typical structure looks like this::

    for epoch in epochs:
        metric.reset()
        for batch_idx, (data, target) in enumerate(train_loader):
            ...
            metric.update(output, target)

        print('Metric value is:', metric.compute())

"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from torch import Tensor


class Metric(ABC):
    """Abstract class for an evaluation metric."""

    DEFAULT_PRECISION = 4

    def __init__(self, accumulate: bool) -> None:
        """
        Create a metric object.

        Args:
            accumulate: whether to accumulate metrics
        """
        self.n_examples = 0
        self.total = 0.0
        self.accumulate = accumulate

    @abstractmethod
    def update(self, output: Tensor, target: Tensor, **kwargs: Any) -> None:
        """
        Update the evaluation metric based on the results of the current batch.

        Args:
            output: the output of the model
            target: the target we want the model to predict
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset metric after every epoch."""
        self.n_examples = 0
        self.total = 0.0

    @abstractmethod
    def compute(self) -> float:
        """
        Compute the overall evaluation metric once everything is done.

        Returns:
            The final evaluation metric as a numeric value.
        """
        raise NotImplementedError


class LossMetric(Metric):
    """A metric for a loss criterion."""

    def __init__(self, criterion: Callable[..., Tensor], accumulate: bool) -> None:
        """
        Create a metric object for computing loss.

        Args:
            criterion: loss function
            accumulate: whether to accumulate metrics
        """
        super(LossMetric, self).__init__(accumulate)
        self.criterion = criterion

    def update(self, output: Tensor, target: Tensor,
               teacher_output: Optional[Tensor] = None, **kwargs: Any) -> None:
        """
        Update the loss metric based on the results of the current batch.

        Args:
            output: the output of the model
            target: the target we want the model to predict
            teacher_output: teacher output for knowledge distillation
        """
        kd_criterion = 0
        if teacher_output is not None:
            kd_criterion = self.criterion(output, teacher_output, target).item()  # type: ignore

        if self.accumulate:
            self.n_examples += output.shape[0]
            if teacher_output is None:
                self.total += self.criterion(output, target, reduction='sum').item()
            else:
                self.total += kd_criterion * output.shape[0]  # kd criterion uses batchmean
        else:
            if teacher_output is None:
                self.total = self.criterion(output, target, reduction='mean').item()
            else:
                self.total = kd_criterion

    def compute(self) -> float:
        """Compute the loss metric once everything is done."""
        return self.total / self.n_examples if self.accumulate else self.total

    def __str__(self) -> str:
        """Get a string representation of the computed metric showing more detailed statistics."""
        return '{0:.{1}f}'.format(self.compute(), 8)


class Top1Accuracy(Metric):
    """Top-1 accuracy metric."""

    def __init__(self, accumulate: bool) -> None:
        """Create a metric object for computing top-1 accuracy."""
        super(Top1Accuracy, self).__init__(accumulate)

    def update(self, output: Tensor, target: Tensor, **kwargs: Any) -> None:
        """
        Update the top-1 accuracy based on the results of the current batch.

        Args:
            output: the output of the model
            target: the target we want the model to predict
        """
        pred_top = output.argmax(dim=1, keepdim=True)
        target = target.view_as(pred_top)
        num_correct = pred_top.eq(target).sum().item()
        if self.accumulate:
            self.n_examples += output.shape[0]
            self.total += num_correct
        else:
            self.n_examples = output.shape[0]
            self.total = num_correct

    def compute(self) -> float:
        """Compute the overall top-1 accuracy once everything is done."""
        return self.total / self.n_examples

    def __str__(self) -> str:
        """Get a string representation of the computed metric showing more detailed statistics."""
        return '{0}/{1} ({2:.{3}f}%)'.format(
            self.total, self.n_examples, 100 * self.compute(), self.DEFAULT_PRECISION
        )


class TopKAccuracy(Metric):
    """Top-K accuracy metric."""

    def __init__(self, k: int, accumulate: bool):
        """
        Create a metric object for computing top-`k` accuracy.

        Args:
            k: The "k" in top-`k` accuracy
            accumulate: whether to accumulate metrics
        """
        super(TopKAccuracy, self).__init__(accumulate)
        self.k = k

    def update(self, output: Tensor, target: Tensor, **kwargs: Any) -> None:
        """
        Update the top-`k` accuracy based on the results of the current batch.

        Args:
            output: the output of the model
            target: the target we want the model to predict
        """
        _, pred_topk = torch.topk(output, dim=1, k=self.k)
        if self.accumulate:
            self.n_examples += output.shape[0]
            self.total += (
                (target.view(-1, 1).expand_as(pred_topk) == pred_topk).sum().item()
            )
        else:
            self.n_examples = output.shape[0]
            self.total = (target.view(-1, 1).expand_as(pred_topk) == pred_topk).sum().item()

    def compute(self) -> float:
        """Compute the overall top-`k` accuracy once everything is done."""
        return self.total / self.n_examples

    def __str__(self) -> str:
        """Get a string representation of the computed metric showing more detailed statistics."""
        return '{0}/{1} ({2:.{3}f}%)'.format(
            self.total, self.n_examples, 100 * self.compute(), self.DEFAULT_PRECISION
        )
