#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Helper functions for calculating optimal binary quantization."""

from typing import Tuple

import torch
import torch.nn.utils.rnn as rnn_utils

from quant.binary.ste import binary_sign


def cost_function(matrix: torch.Tensor, v1s: torch.Tensor, ternary: bool = False) -> torch.Tensor:
    """
    Compute the cost function to find the optimal v1.

    The cost function is equation (8) in the paper, for k=2.
    It can be derived by expanding s1, s2 using the foldable quantization equation (9).

    Args:
        matrix: original 2D tensor
        v1s: 2D tensor containing potential optimal solutions
        ternary: compute cost for ternary function

    Returns:
        Norms as a 2D tensor
    """
    matrix_view = matrix.view(matrix.shape[0], 1, -1)
    v1s_view = v1s.view(v1s.shape[0], v1s.shape[1], 1)
    s2_arg = matrix_view - v1s_view * binary_sign(matrix_view)
    if ternary:
        v2 = v1s_view
    else:
        v2 = s2_arg.abs().mean(dim=-1, keepdim=True)
    return torch.norm(s2_arg - v2 * binary_sign(s2_arg), dim=-1)  # type: ignore


def compute_mask(matrix: torch.Tensor, ternary: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mask for a 2D tensor of absolute values.

    The mask reveals potential optimal values.

    Args:
        matrix: A 2D tensor of absolute values.
        ternary: whether we are computing mask for ternary algorithm

    Returns:
        A 2-tuple of tensors, where the first element is a mask
        tensor and the second element are values selected
    """
    values, _ = torch.sort(matrix, dim=1)
    cum_sums = values.cumsum(dim=1)

    # store counts of elements at the corresponding position
    counts = torch.arange(1, matrix.shape[1] + 1, device=matrix.device)
    counts_rev = torch.flip(counts, [0]) - 1
    counts_rev[-1] = 1  # avoid division by 0, value at this pos. will not be used

    m1s = None
    if not ternary:
        # m1s stores cumulative means from left to right (chopping left and right most values)
        m1s = (cum_sums / counts)[:, 1:-1]
    # m2s stores cumulative means from right to left (chopping left and right most values)
    m2s = ((cum_sums[:, -1:] - cum_sums) / counts_rev)[:, 1:-1]

    # re-using m1s and m2s to save memory
    # using m1s and m2s values to find potential optimal solutions to v1 and v2
    if not ternary:
        m1s = 0.5 * (m1s + m2s)
    m2s = 0.5 * m2s
    # Find potential solutions in inner region and boundary
    # Instead of finding equality, find index where m1s or m2s
    # is >= than everything on the left and <= than everything on the right
    mask = (values[:, 1:-1] <= m2s) * (m2s <= values[:, 2:])
    if not ternary:
        mask = mask + (values[:, 1:-1] <= m1s) * (m1s <= values[:, 2:])

    masked_vs = torch.masked_select(values[:, 1:-1], mask)
    return mask, masked_vs


def _handle_ternary_min_gt_half_avg(
    matrix: torch.Tensor, masked_vs: torch.Tensor, split_sizes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Handle edge case in ternary case when min value is less than half of average."""
    # Suppose x is the absolute value of the tensor to be quantized
    # For least squares, 2 bits, the optimal v will always be between min(x) and max(x)
    # For the ternary case, the optimal v could be < min(x)
    # This occurs when min(x) > 1/2 * avg(x)
    # When this occurs, then we should append 1/2 * avg(x) as a solution
    rows_mean = matrix.mean(dim=1)
    rows_min, _ = matrix.min(dim=1)
    row_min_gt_half_avg = rows_min > 0.5 * rows_mean

    if not torch.any(row_min_gt_half_avg):
        # This should almost always be the case
        return masked_vs, split_sizes

    # This should rarely happen if at all (e.g., when all elements are equal)
    new_masked_vs = []
    masked_vs_list = masked_vs.tolist()
    current_pos = 0
    for i, v in enumerate(row_min_gt_half_avg):
        if split_sizes[i] > 0:
            new_masked_vs.extend(
                masked_vs_list[current_pos:current_pos + int(split_sizes[i].item())]
            )
            current_pos += int(split_sizes[i].item())

        if v:
            split_sizes[i] += 1
            new_masked_vs.append(rows_mean[i].item() / 2)

    return torch.tensor(new_masked_vs, device=matrix.device), split_sizes


def opt_v1(matrix: torch.Tensor, ternary: bool, skip: int = 1) -> torch.Tensor:  # type: ignore
    """
    Implement the algorithm to find v1 for least squares 2-bit and ternary algorithm.

    Args:
        matrix: A 2D tensor
        ternary: whether to do ternary optimization
        skip: increment in potential solution space to speed up computation

    Returns:
        Optimal v1
    """
    with torch.no_grad():
        matrix_skipped = matrix[..., ::skip].abs()
        mask, masked_vs = compute_mask(matrix_skipped, ternary)

        # masked_vs is a vector, we need to separate it into potential
        # optimal solutions by row (dim 0)
        split_sizes = mask.sum(dim=1)

        if ternary:
            # handle a special case for ternary that rarely occurs
            masked_vs, split_sizes = _handle_ternary_min_gt_half_avg(
                matrix_skipped, masked_vs, split_sizes
            )

        vs = torch.split(masked_vs, split_sizes.tolist())  # type: ignore
        vs = rnn_utils.pad_sequence(vs, batch_first=True)  # type: ignore

        costs = cost_function(matrix_skipped, vs, ternary)
        indices = torch.argmin(costs, dim=-1, keepdim=True)

        v1 = torch.gather(vs, 1, indices)

        return v1
