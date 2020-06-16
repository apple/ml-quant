#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test activation quantization classes."""

import torch

import quant.binary.quantization as quantization
from quant.binary.activation_quantization import ActivationQuantizerLS1,\
    ActivationQuantizerLS2, ActivationQuantizerLST, ActivationQuantizerGF


def test_activation_quantizer_ls1_no_ma():
    """Test no moving average mode of activation quantizer for least squares 1 bit."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor

    quantizer_ls1_no_ma = ActivationQuantizerLS1('off')
    quantizer_ls1_no_ma.train()
    quantizer_ls1_no_ma(x)  # v1 should be 2 for all examples
    x_q_train_no_ma = quantizer_ls1_no_ma(x)  # call twice so moving avg changes if used
    assert torch.all(x_q_train_no_ma == 2.0)

    quantizer_ls1_no_ma.eval()
    x_q_eval_no_ma = quantizer_ls1_no_ma(x2)
    # v1 should not be cached, so it should be recomputed
    _, expected = quantization.quantizer_ls_1(x2)
    assert torch.all(x_q_eval_no_ma.eq(expected))
    assert not torch.all(x_q_eval_no_ma.eq(x_q_train_no_ma))


def test_activation_quantizer_ls1_eval_only():
    """Test eval_only mode of activation quantizer for least squares 1 bit."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_ls1_eval_only = ActivationQuantizerLS1('eval_only', 0.9)
    quantizer_ls1_eval_only.train()
    x_q_train_eval_only = quantizer_ls1_eval_only(x)
    assert torch.all(x_q_train_eval_only == 2.0)
    x_q_train_eval_only = quantizer_ls1_eval_only(x3)
    assert torch.all(x_q_train_eval_only == 4.0)

    quantizer_ls1_eval_only.eval()
    x_q_eval_eval_only = quantizer_ls1_eval_only(x2)
    # moving average should cause v1 to become 2 * 0.9 + 4 * 0.1 = 2.2
    assert torch.all(x_q_eval_eval_only == 2.2)


def test_activation_quantizer_ls1_train_and_eval():
    """Test train_and_eval mode of activation quantizer for least squares 1 bit."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_ls1_all_ma = ActivationQuantizerLS1('train_and_eval', 0.9)

    quantizer_ls1_all_ma.train()
    x_q_train_all_ma = quantizer_ls1_all_ma(x)
    assert torch.all(x_q_train_all_ma == 2.0)
    x_q_train_all_ma = quantizer_ls1_all_ma(x3)
    # 2 * 0.9 + 4 * 0.1 = 2.2
    assert torch.all(x_q_train_all_ma == 2.2)

    quantizer_ls1_all_ma.eval()
    x_q_eval_all_ma = quantizer_ls1_all_ma(x2)
    assert torch.all(x_q_eval_all_ma == 2.2)


def test_activation_quantizer_ls2_no_ma():
    """Test no moving average mode of activation quantizer for least squares 2 bits."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor

    quantizer_ls2_no_ma = ActivationQuantizerLS2('off')
    quantizer_ls2_no_ma.train()
    quantizer_ls2_no_ma(x)  # v1 should be 2 for all examples
    x_q_train_no_ma = quantizer_ls2_no_ma(x)  # call twice so moving avg changes if used
    assert torch.all(x_q_train_no_ma == 2.0)

    quantizer_ls2_no_ma.eval()
    x_q_eval_no_ma = quantizer_ls2_no_ma(x2)
    # v1, v2 should not be cached, so it should be recomputed
    _, _, expected = quantization.quantizer_ls_2(x2)
    assert torch.all(x_q_eval_no_ma.eq(expected))
    assert not torch.all(x_q_eval_no_ma.eq(x_q_train_no_ma))


def test_activation_quantizer_ls2_eval_only():
    """Test eval_only mode of activation quantizer for least squares 2 bits."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_ls2_eval_only = ActivationQuantizerLS2('eval_only', 0.9)
    quantizer_ls2_eval_only.train()
    x_q_train_eval_only = quantizer_ls2_eval_only(x)
    assert torch.all(x_q_train_eval_only == 2.0)
    x_q_train_eval_only = quantizer_ls2_eval_only(x3)
    assert torch.all(x_q_train_eval_only == 4.0)

    quantizer_ls2_eval_only.eval()
    x_q_eval_eval_only = quantizer_ls2_eval_only(x2)
    # moving average should cause v1 to become 2 * 0.9 + 4 * 0.1 = 2.2, v2 should be 0
    torch.all(x_q_eval_eval_only == 2.2)


def test_activation_quantizer_ls2_train_and_eval():
    """Test train_and_eval mode of activation quantizer for least squares 2 bits."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_ls2_all_ma = ActivationQuantizerLS2('train_and_eval', 0.9)

    quantizer_ls2_all_ma.train()
    x_q_train_all_ma = quantizer_ls2_all_ma(x)
    assert torch.all(x_q_train_all_ma == 2.0)
    x_q_train_all_ma = quantizer_ls2_all_ma(x3)
    # v1 = 2 * 0.9 + 4 * 0.1 = 2.2, v2 should be 0
    assert torch.all(x_q_train_all_ma == 2.2)

    quantizer_ls2_all_ma.eval()
    x_q_eval_all_ma = quantizer_ls2_all_ma(x2)
    assert torch.all(x_q_eval_all_ma == 2.2)


def test_activation_quantizer_lsT_no_ma():
    """Test no moving average mode of activation quantizer for least squares ternary."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor

    quantizer_lsT_no_ma = ActivationQuantizerLST('off')
    quantizer_lsT_no_ma.train()
    quantizer_lsT_no_ma(x)
    x_q_train_no_ma = quantizer_lsT_no_ma(x)  # call twice so moving avg changes if used
    assert torch.all(x_q_train_no_ma == 2.0)

    quantizer_lsT_no_ma.eval()
    x_q_eval_no_ma = quantizer_lsT_no_ma(x2)
    # v1 should not be cached, so it should be recomputed
    _, expected = quantization.quantizer_ls_ternary(x2)
    assert torch.all(x_q_eval_no_ma.eq(expected))


def test_activation_quantizer_lsT_eval_only():
    """Test eval_only mode of activation quantizer for least squares ternary."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_lsT_eval_only = ActivationQuantizerLST('eval_only', 0.9)
    quantizer_lsT_eval_only.train()
    # moving average should cause tracked v1 to become 1.0 after call
    x_q_train_eval_only = quantizer_lsT_eval_only(x)
    assert torch.all(x_q_train_eval_only == 2.0)
    # moving average should cause tracked v1 to become 1 * 0.9 + 2 * 0.1 = 1.1 after call
    x_q_train_eval_only = quantizer_lsT_eval_only(x3)
    assert torch.all(x_q_train_eval_only == 4.0)

    quantizer_lsT_eval_only.eval()
    x_q_eval_eval_only = quantizer_lsT_eval_only(x2)
    _, expected = quantization.quantizer_ls_ternary(x2, torch.tensor([1.1] * 32))
    assert torch.all(x_q_eval_eval_only.eq(expected))


def test_activation_quantizer_lsT_train_and_eval():
    """Test train_and_eval mode of activation quantizer for least squares ternary."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_lsT_all_ma = ActivationQuantizerLST('train_and_eval', 0.9)

    quantizer_lsT_all_ma.train()
    # moving average should cause tracked v1 to become 1.0 after call
    x_q_train_all_ma = quantizer_lsT_all_ma(x)
    _, expected = quantization.quantizer_ls_ternary(x, torch.tensor([1.0] * 32))
    assert torch.all(x_q_train_all_ma.eq(expected))
    # moving average should cause tracked v1 to become 1 * 0.9 + 2 * 0.1 = 1.1 after call
    x_q_train_all_ma = quantizer_lsT_all_ma(x3)
    _, expected = quantization.quantizer_ls_ternary(x, torch.tensor([1.1] * 32))
    assert torch.all(x_q_train_all_ma.eq(expected))

    quantizer_lsT_all_ma.eval()
    x_q_eval_train_and_eval = quantizer_lsT_all_ma(x2)
    _, expected = quantization.quantizer_ls_ternary(x2, torch.tensor([1.1] * 32))
    assert torch.all(x_q_eval_train_and_eval.eq(expected))


def test_activation_quantizer_gf_no_ma():
    """Test no moving average mode of activation quantizer for greedy foldable."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.ones(32, 16, 3, 3) * 4

    quantizer_gf_no_ma = ActivationQuantizerGF(2, 'off')
    quantizer_gf_no_ma.train()
    quantizer_gf_no_ma(x)
    x_q_train_no_ma = quantizer_gf_no_ma(x)  # call twice so moving avg changes if used
    assert torch.all(x_q_train_no_ma == 2.0)

    quantizer_gf_no_ma.eval()
    x_q_eval_no_ma = quantizer_gf_no_ma(x2)
    assert torch.all(x_q_eval_no_ma == 4.0)


def test_activation_quantizer_gf_eval_only():
    """Test eval_only mode of activation quantizer for greedy foldable."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_gf_eval_only = ActivationQuantizerGF(2, 'eval_only', 0.9)
    quantizer_gf_eval_only.train()
    x_q_train_eval_only = quantizer_gf_eval_only(x)
    assert torch.all(x_q_train_eval_only == 2.0)
    x_q_train_eval_only = quantizer_gf_eval_only(x3)
    assert torch.all(x_q_train_eval_only == 4.0)

    quantizer_gf_eval_only.eval()
    x_q_eval_eval_only = quantizer_gf_eval_only(x2)
    # moving average should cause v1 to become 2 * 0.9 + 4 * 0.1 = 2.2, v2 should be 0
    torch.all(x_q_eval_eval_only == 2.2)


def test_activation_quantizer_gf_train_and_eval():
    """Test train_and_eval mode of activation quantizer for least squares greedy foldable."""
    torch.manual_seed(1234)
    x = torch.ones(32, 16, 3, 3) * 2
    x2 = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    x3 = torch.ones(32, 16, 3, 3) * 4

    quantizer_gf_all_ma = ActivationQuantizerGF(2, 'train_and_eval', 0.9)

    quantizer_gf_all_ma.train()
    x_q_train_all_ma = quantizer_gf_all_ma(x)
    assert torch.all(x_q_train_all_ma == 2.0)
    x_q_train_all_ma = quantizer_gf_all_ma(x3)
    # v1 = 2 * 0.9 + 4 * 0.1 = 2.2, v2 should be 0
    assert torch.all(x_q_train_all_ma == 2.2)

    quantizer_gf_all_ma.eval()
    x_q_eval_all_ma = quantizer_gf_all_ma(x2)
    assert torch.all(x_q_eval_all_ma == 2.2)
