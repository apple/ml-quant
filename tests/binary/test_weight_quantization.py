#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Test weight quantization functions classes."""

import torch

import quant.binary.quantization as quantization
import quant.binary.weight_quantization as weight_quantization


def test_weight_quantizer_ls1_modes():
    """Test training mode and eval mode for WeightQuantizerLS1."""
    torch.manual_seed(1234)
    quantizer_ls1 = weight_quantization.WeightQuantizerLS1(32)
    w = torch.ones(32, 16, 3, 3) * 2

    quantizer_ls1.train()
    w_q_train = quantizer_ls1(w)  # v1 should be 2 for all channels
    assert torch.all(w_q_train == 2.0)

    quantizer_ls1.eval()
    w = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    w_q_eval = quantizer_ls1(w)

    # since every element of matrix is quantized to +1, and scaling factor is 2
    assert torch.all(w_q_train.eq(w_q_eval))


def test_weight_quantizer_ls2_modes():
    """Test training mode and eval mode for WeightQuantizerLS2."""
    torch.manual_seed(1234)
    quantizer_ls2 = weight_quantization.WeightQuantizerLS2(32)
    w = torch.ones(32, 16, 3, 3) * 2

    quantizer_ls2.train()
    w_q_train = quantizer_ls2(w)
    assert torch.all(w_q_train == 2.0)

    quantizer_ls2.eval()
    w = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    w_q_eval = quantizer_ls2(w)

    assert torch.all(w_q_train.eq(w_q_eval))


def test_weight_quantizer_lsT_modes():
    """Test training mode and eval mode for WeightQuantizerLST."""
    torch.manual_seed(1234)
    quantizer_lsT = weight_quantization.WeightQuantizerLST(32)
    w = torch.rand(32, 16, 3, 3)

    quantizer_lsT.train()
    _ = quantizer_lsT(w)
    v1 = quantizer_lsT.v1

    quantizer_lsT.eval()
    w = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    w_q_eval = quantizer_lsT(w)
    _, w_q_eval_expected = quantization.quantizer_ls_ternary(w, v1=v1)

    assert torch.all(w_q_eval.eq(w_q_eval_expected))


def test_weight_quantizer_gf_modes():
    """Test training mode and eval mode for WeightQuantizerGF."""
    torch.manual_seed(1234)
    quantizer_gf = weight_quantization.WeightQuantizerGF(32, 2)
    w = torch.ones(32, 16, 3, 3) * 2

    quantizer_gf.train()
    w_q_train = quantizer_gf(w)
    assert torch.all(w_q_train == 2.0)

    quantizer_gf.eval()
    w = torch.rand(32, 16, 3, 3)  # some random, but all positive tensor
    w_q_eval = quantizer_gf(w)

    assert torch.all(w_q_train.eq(w_q_eval))
