#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Criterion for knowledge distillation."""

import torch
import torch.nn.functional as F


def kd_criterion(
    output_student: torch.Tensor,
    output_teacher: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    freeze_teacher: bool = True,
    teacher_correction: bool = True,
) -> torch.Tensor:
    """
    Criterion for knowledge distillation.

    Args:
        output_student: student network output
        output_teacher: teacher network output
        target: target tensor
        temperature: temperature
        freeze_teacher: whether to freeze teacher
        teacher_correction: whether to use the regular loss when the teacher's prediction
            is different from the true label for that particular example

    Returns:
        loss based on knowledge distillation criterion
    """
    output_teacher_val = output_teacher.detach() if freeze_teacher else output_teacher

    kd_loss = F.kl_div(
        F.log_softmax(output_student / temperature, dim=1),
        F.softmax(output_teacher_val / temperature, dim=1),
        reduction='none'
    ) * (temperature * temperature)
    kd_loss = kd_loss.sum(dim=1)

    if teacher_correction:
        pred_teacher = output_teacher_val.argmax(dim=1)
        correct_mask = pred_teacher.eq(pred_teacher)
        ce_loss = F.cross_entropy(output_student, target, reduction='none')
        total_loss = correct_mask * kd_loss + ~correct_mask * ce_loss
    else:
        total_loss = kd_loss

    return total_loss.mean()
