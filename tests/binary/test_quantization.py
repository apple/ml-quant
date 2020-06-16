#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test quantization functions."""

import torch

import quant.binary.quantization as quantization
from quant.binary.ste import binarize, binary_sign


def test_clamp_identity():
    """Test identity clamp function."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    assert torch.all(x.eq(quantization.clamp_identity(x)))


def test_clamp_symmetric():
    """Test symmetric clamp function."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    assert torch.all(torch.tensor([-1, 0, 1, 1]).eq(quantization.clamp_symmetric(x, 1)))
    assert torch.all(torch.tensor([-0.5, 0, 0.5, 0.5]).eq(quantization.clamp_symmetric(x, 0.5)))
    assert torch.all(torch.tensor([-1, 0, 1, 2]).eq(quantization.clamp_symmetric(x, 2)))
    assert torch.all(torch.tensor([-1, 0, 1, 2]).eq(quantization.clamp_symmetric(x, 3)))


def test_quantizer_fp():
    """Test full precision (identity) quantizer."""
    quantizer_fp = quantization.QuantizerFP()
    x = torch.tensor([-1, 0, 1, 2])
    assert torch.all(x.eq(quantizer_fp(x)))


def test_quantizer_ls_1_optimal():
    """Test 1-bit optimal least-squares scaled binary quantization."""
    torch.manual_seed(1234)
    x = torch.randn(1000, 3, 64, 64)

    _, x_q = quantization.quantizer_ls_1(x)
    assert x_q.shape == x.shape

    # Check x_q has lower least-squares error compared with using random scaling factors
    subopt_scaling_factor = torch.randn(1000, 1, 1, 1).abs()
    subopt_quantization = subopt_scaling_factor * binarize(x)
    opt_costs = torch.norm((x_q - x).view(1000, -1), dim=1)
    subopt_costs = torch.norm((subopt_quantization - x).view(1000, -1), dim=1)
    assert torch.all(opt_costs <= subopt_costs)


def test_quantizer_ls_2_optimal():
    """Test 2-bit optimal least squares scaled binary quantization."""
    torch.manual_seed(1234)
    x = torch.randn(1000, 3, 64, 64)

    _, _, x_q = quantization.quantizer_ls_2(x, skip=1)
    assert x_q.shape == x.shape

    # Check x_q has lower least-squares error compared with using random scaling factors
    rand_indices = torch.randint(0, 3 * 64 * 64, (1000,))
    subopt_v1 = x.view(1000, -1)[torch.arange(1000), rand_indices].view(1000, 1).abs()
    s2 = x.view(1000, -1) - subopt_v1 * binary_sign(x.view(1000, -1))
    subopt_v2 = s2.abs().mean(dim=-1, keepdim=True)

    b1 = binarize(x)
    subopt_v1 = subopt_v1.view(1000, 1, 1, 1)
    subopt_v2 = subopt_v2.view(1000, 1, 1, 1)
    subopt_quantization = subopt_v1 * b1 + subopt_v2 * binarize(x - subopt_v1 * b1)

    opt_costs = torch.norm((x_q - x).view(1000, -1), dim=1)
    subopt_costs = torch.norm((subopt_quantization - x).view(1000, -1), dim=1)
    assert torch.all(opt_costs <= subopt_costs)


def test_quantizer_ls_T_optimal():
    """Test ternary optimal least squares scaled binary quantization."""
    torch.manual_seed(1234)
    x = torch.randn(1000, 3, 64, 64)

    _, x_q = quantization.quantizer_ls_ternary(x, skip=1)
    assert x_q.shape == x.shape

    # Check x_q has lower least-squares error compared with using random scaling factors
    rand_indices = torch.randint(0, 3 * 64 * 64, (1000,))
    subopt_v1 = x.view(1000, -1)[torch.arange(1000), rand_indices].view(1000, 1, 1, 1).abs()
    b1 = binarize(x)
    subopt_quantization = subopt_v1 * b1 + subopt_v1 * binarize(x - subopt_v1 * b1)

    opt_costs = torch.norm((x_q - x).view(1000, -1), dim=1)
    subopt_costs = torch.norm((subopt_quantization - x).view(1000, -1), dim=1)
    assert torch.all(opt_costs <= subopt_costs)


def test_quantizer_ls_T_all_inputs_equal():
    """Test ternary optimal least squares scaled binary quantization edge case."""
    torch.manual_seed(1234)
    x = torch.ones(32, 3, 16, 16) * 2
    _, x_q = quantization.quantizer_ls_ternary(x)

    assert torch.all(x_q == 2.0)

    # Test the case just certain rows have all elements equal
    x = torch.rand(32, 3, 16, 16)
    x[1, :, :, :] = torch.ones(3, 16, 16) * 2
    x[9, :, :, :] = torch.ones(3, 16, 16) * -3

    _, x_q = quantization.quantizer_ls_ternary(x)

    assert torch.all(x_q[1, :, :, :] == 2)
    assert torch.all(x_q[9, :, :, :] == -3)


def test_quantizer_gf_more_bits_are_better():
    """Test the more bits are used for gf, the better it is."""
    torch.manual_seed(1234)
    x = torch.randn(1000, 3, 64, 64)

    _, x_q_gf1 = quantization.quantizer_gf(x, k=1)
    _, x_q_gf2 = quantization.quantizer_gf(x, k=2)
    _, x_q_gf3 = quantization.quantizer_gf(x, k=3)
    _, x_q_gf4 = quantization.quantizer_gf(x, k=4)

    gf1_costs = torch.norm((x_q_gf1 - x).view(1000, -1), dim=1)
    gf2_costs = torch.norm((x_q_gf2 - x).view(1000, -1), dim=1)
    gf3_costs = torch.norm((x_q_gf3 - x).view(1000, -1), dim=1)
    gf4_costs = torch.norm((x_q_gf4 - x).view(1000, -1), dim=1)

    assert torch.all(gf2_costs <= gf1_costs)
    assert torch.all(gf3_costs <= gf2_costs)
    assert torch.all(gf4_costs <= gf3_costs)


def test_quantizer_ls2_better_than_lsT():
    """Test ls-2 is better than ls-T, which is better than ls-1."""
    torch.manual_seed(1234)
    x = torch.randn(1000, 3, 64, 64)

    _, _, x_q_ls2 = quantization.quantizer_ls_2(x, skip=1)
    _, x_q_lsT = quantization.quantizer_ls_ternary(x, skip=1)
    _, x_q_ls1 = quantization.quantizer_ls_1(x)

    ls2_costs = torch.norm((x_q_ls2 - x).view(1000, -1), dim=1)
    lsT_costs = torch.norm((x_q_lsT - x).view(1000, -1), dim=1)
    ls1_costs = torch.norm((x_q_ls1 - x).view(1000, -1), dim=1)

    assert torch.all(ls2_costs <= lsT_costs)
    assert torch.all(lsT_costs <= ls1_costs)


def test_quantizer_ls2_better_than_gf2():
    """Test ls-2 is better than gf-2, which is better than ls-1."""
    torch.manual_seed(1234)
    x = torch.randn(1000, 3, 64, 64)

    _, _, x_q_ls2 = quantization.quantizer_ls_2(x, skip=1)
    _, x_q_gf2 = quantization.quantizer_gf(x, k=2)
    _, x_q_ls1 = quantization.quantizer_ls_1(x)

    ls2_costs = torch.norm((x_q_ls2 - x).view(1000, -1), dim=1)
    gf2_costs = torch.norm((x_q_gf2 - x).view(1000, -1), dim=1)
    ls1_costs = torch.norm((x_q_ls1 - x).view(1000, -1), dim=1)

    assert torch.all(ls2_costs <= gf2_costs)
    assert torch.all(gf2_costs <= ls1_costs)
