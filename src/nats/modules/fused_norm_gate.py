# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard


@triton.heuristics({
    'HAS_X_WEIGHTS': lambda args: args['x_weights_after_norm'] is not None,
    'STORE_RESIDUAL_OUT': lambda args: args['residual_out1'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_WEIGHT1': lambda args: args['w1'] is not None,
    'HAS_BIAS1': lambda args: args['b1'] is not None,
    'HAS_WEIGHT2': lambda args: args['w2'] is not None,
    'HAS_BIAS2': lambda args: args['b2'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=['D', 'NB', 'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs
)
@triton.jit
def multi_input_layer_norm_gated_fwd_kernel(
        x1,  # pointer to the first input
        x2,  # pointer to the second input
        g,  # pointer to the gate
        y,  # pointer to the output
        x_weights_after_norm, # pointer to the output norm,
        w1,  # pointer to the weights1
        w2,  # pointer to the weights2
        b1,  # pointer to the biases 1
        b2,  # pointer to the biases 2
        residual,  # pointer to the residual
        residual_out1,  # pointer to the first residual
        residual_out2,  # pointer to the second residual
        mean1,  # pointer to the first mean
        mean2,  # pointer to the second mean
        rstd1,  # pointer to the first 1/std
        rstd2,  # pointer to the second 1/std
        eps,  # epsilon to avoid division by zero
        T,  # number of rows in x
        D: tl.constexpr,  # number of columns in x
        BT: tl.constexpr,
        BD: tl.constexpr,
        NB: tl.constexpr,
        ACTIVATION: tl.constexpr,
        IS_RMS_NORM: tl.constexpr,
        STORE_RESIDUAL_OUT: tl.constexpr,
        HAS_X_WEIGHTS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        HAS_WEIGHT1: tl.constexpr,
        HAS_BIAS1: tl.constexpr,
        HAS_WEIGHT2: tl.constexpr,
        HAS_BIAS2: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x1 = tl.make_block_ptr(x1, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x1 = tl.load(p_x1, boundary_check=(0, 1)).to(tl.float32)

    p_x2 = tl.make_block_ptr(x2, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x2 = tl.load(p_x2, boundary_check=(0, 1)).to(tl.float32)

    if HAS_X_WEIGHTS:
        p_x1_w = tl.make_block_ptr(x_weights_after_norm, (T, 2), (2, 1), (i_t * BT, 0), (BT, 1), (1,0))
        p_x2_w = tl.make_block_ptr(x_weights_after_norm, (T, 2), (2, 1), (i_t * BT, 1), (BT, 1), (1,0))

        b_x1_w = tl.load(p_x1_w, boundary_check=(0, 1)).to(tl.float32)
        b_x2_w = tl.load(p_x2_w, boundary_check=(0,1)).to(tl.float32)

    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        b_res = tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
        b_x1 += b_res
        b_x2 += b_res


    if STORE_RESIDUAL_OUT:
        p_res_out1 = tl.make_block_ptr(residual_out1, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out1, b_x1.to(p_res_out1.dtype.element_ty), boundary_check=(0, 1))

        p_res_out2 = tl.make_block_ptr(residual_out2, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out2, b_x1.to(p_res_out2.dtype.element_ty), boundary_check=(0, 1))

    if not IS_RMS_NORM:
        b_mean1 = tl.sum(b_x1, axis=1) / D
        p_mean1 = tl.make_block_ptr(mean1, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean1, b_mean1.to(p_mean1.dtype.element_ty), boundary_check=(0,))
        b_xbar1 = tl.where(m_d[None, :], b_x1 - b_mean1[:, None], 0.0)
        b_var1 = tl.sum(b_xbar1 * b_xbar1, axis=1) / D

        b_mean2 = tl.sum(b_x2, axis=1) / D
        p_mean2 = tl.make_block_ptr(mean2, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean2, b_mean2.to(p_mean2.dtype.element_ty), boundary_check=(0,))
        b_xbar2 = tl.where(m_d[None, :], b_x2 - b_mean2[:, None], 0.0)
        b_var2 = tl.sum(b_xbar2 * b_xbar2, axis=1) / D
    else:
        b_xbar1 = tl.where(m_d[None, :], b_x1, 0.0)
        b_var1 = tl.sum(b_xbar1 * b_xbar1, axis=1) / D

        b_xbar2 = tl.where(m_d[None, :], b_x2, 0.0)
        b_var2 = tl.sum(b_xbar2 * b_xbar2, axis=1) / D

    b_rstd1 = 1 / tl.sqrt(b_var1 + eps)
    b_rstd2 = 1 / tl.sqrt(b_var2 + eps)

    p_rstd1 = tl.make_block_ptr(rstd1, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd1, b_rstd1.to(p_rstd1.dtype.element_ty), boundary_check=(0,))

    p_rstd2 = tl.make_block_ptr(rstd2, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd2, b_rstd2.to(p_rstd2.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT1:
        b_w1 = tl.load(w1 + o_d, mask=m_d).to(tl.float32)
    if HAS_WEIGHT2:
        b_w2 = tl.load(w2 + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS1:
        b_b1 = tl.load(b1 + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS2:
        b_b2 = tl.load(b2 + o_d, mask=m_d).to(tl.float32)

    b_x1_hat = (b_x1 - b_mean1[:, None]) * b_rstd1[:, None] if not IS_RMS_NORM else b_x1 * b_rstd1[:, None]
    b_x2_hat = (b_x2 - b_mean2[:, None]) * b_rstd2[:, None] if not IS_RMS_NORM else b_x2 * b_rstd2[:, None]
    # if we only have one set of weights, then they are first added and then
    HAS_ONLY_WEIGHT1 = HAS_WEIGHT1 and not HAS_WEIGHT2
    HAS_BOTH_WEIGHTS = HAS_WEIGHT1 and HAS_WEIGHT2
    if HAS_ONLY_WEIGHT1:
        if HAS_X_WEIGHTS:
            b_x_hat = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w
        else:
            b_x_hat = (b_x1_hat + b_x2_hat)
        b_y = b_x_hat * b_w1[None, :] if HAS_WEIGHT1 else b_x_hat
        if HAS_BIAS1:
            b_y = b_y + b_b1[None, :]
    elif HAS_BOTH_WEIGHTS:
        b_x1_hat = b_x1_hat * b_w1[None, :]
        b_x2_hat = b_x2_hat * b_w2[None, :]
        if HAS_BIAS1:
            b_x1_hat = b_x1_hat + b_b1[None, :]
        if HAS_BIAS2:
            b_x2_hat = b_x2_hat + b_b2[None, :]
        if HAS_X_WEIGHTS:
            b_y = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w
    else:
        # we set by default that if weight1 does not exist, weights2 will be ignored
        if HAS_BIAS1:
            b_x1_hat = b_x1_hat + b_b1[None, :]
        if HAS_BIAS2:
            b_x2_hat = b_x2_hat + b_b2[None, :]
        b_y = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w if HAS_X_WEIGHTS else b_x1_hat + b_x2_hat

    # swish/sigmoid output gate
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == 'swish':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_X_WEIGHTS': lambda args: args['x_weights_after_norm'] is not None,
    'STORE_RESIDUAL_OUT': lambda args: args['residual_out1'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_WEIGHT1': lambda args: args['w1'] is not None,
    'HAS_BIAS1': lambda args: args['b1'] is not None,
    'HAS_WEIGHT2': lambda args: args['w2'] is not None,
    'HAS_BIAS2': lambda args: args['b2'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['D', 'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs
)
@triton.jit
def multi_input_layer_norm_gated_fwd_kernel1(
        x1,  # pointer to the first input
        x2,  # pointer to the second input
        g,  # pointer to the gate
        y,  # pointer to the output
        x_weights_after_norm,  # pointer to the output norm,
        w1,  # pointer to the first weights
        b1,  # pointer to the first biases
        w2,  # pointer to the second weights
        b2,  # pointer to the second biases
        residual,  # pointer to the residual
        residual_out1,  # pointer to the residual
        residual_out2,  # pointer to the second residual
        mean1,  # pointer to the first mean
        rstd1,  # pointer to the first 1/std
        mean2,  # pointer to the second mean
        rstd2,  # pointer to the second 1/std
        eps,  # epsilon to avoid division by zero
        D: tl.constexpr,  # number of columns in x
        BD: tl.constexpr,
        ACTIVATION: tl.constexpr,
        IS_RMS_NORM: tl.constexpr,
        STORE_RESIDUAL_OUT: tl.constexpr,
        HAS_X_WEIGHTS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        HAS_WEIGHT1: tl.constexpr,
        HAS_BIAS1: tl.constexpr,
        HAS_WEIGHT2: tl.constexpr,
        HAS_BIAS2: tl.constexpr,
):
    i_t = tl.program_id(0)
    x1 += i_t * D
    x2 += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out1 += i_t * D
        residual_out2 += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x1 = tl.load(x1 + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_x2 = tl.load(x2 + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_res = tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_x1 += b_res
        b_x2 += b_res
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out1 + o_d, b_x1, mask=m_d)
        tl.store(residual_out2 + o_d, b_x2, mask=m_d)

    if HAS_X_WEIGHTS:
        x_weights_after_norm += i_t * 2
        b_x1_w = tl.load(x_weights_after_norm)
        b_x2_w = tl.load(x_weights_after_norm + 1)

    if not IS_RMS_NORM:
        b_mean1 = tl.sum(b_x1, axis=0) / D
        b_mean2 = tl.sum(b_x2, axis=0) / D
        tl.store(mean1 + i_t, b_mean1)
        tl.store(mean2 + i_t, b_mean2)
        b_xbar1 = tl.where(m_d, b_x1 - b_mean1, 0.0)
        b_var1 = tl.sum(b_xbar1 * b_xbar1, axis=0) / D
        b_xbar2 = tl.where(m_d, b_x2 - b_mean2, 0.0)
        b_var2 = tl.sum(b_xbar2 * b_xbar2, axis=0) / D
    else:
        b_xbar1 = tl.where(m_d, b_x1, 0.0)
        b_var1 = tl.sum(b_xbar1 * b_xbar1, axis=0) / D
        b_xbar2 = tl.where(m_d, b_x2, 0.0)
        b_var2 = tl.sum(b_xbar2 * b_xbar2, axis=0) / D

    b_rstd1 = 1 / tl.sqrt(b_var1 + eps)
    tl.store(rstd1 + i_t, b_rstd1)

    b_rstd2 = 1 / tl.sqrt(b_var2 + eps)
    tl.store(rstd2 + i_t, b_rstd2)

    if HAS_WEIGHT1:
        b_w1 = tl.load(w1 + o_d, mask=m_d).to(tl.float32)
    if HAS_WEIGHT2:
        b_w2 = tl.load(w2 + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS1:
        b_b1 = tl.load(b1 + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS2:
        b_b2 = tl.load(b2 + o_d, mask=m_d).to(tl.float32)

    b_x1_hat = (b_x1 - b_mean1) * b_rstd1 if not IS_RMS_NORM else b_x1 * b_rstd1
    b_x2_hat = (b_x2 - b_mean2) * b_rstd2 if not IS_RMS_NORM else b_x2 * b_rstd2
    HAS_ONLY_WEIGHT1 = HAS_WEIGHT1 and not HAS_WEIGHT2
    HAS_BOTH_WEIGHTS = HAS_WEIGHT1 and HAS_WEIGHT2

    if HAS_ONLY_WEIGHT1:
        if HAS_X_WEIGHTS:
            b_x_hat = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w
        else:
            b_x_hat = (b_x1_hat + b_x2_hat)
        b_y = b_x_hat * b_w1 if HAS_WEIGHT1 else b_x_hat
        if HAS_BIAS1:
            b_y = b_y + b_b1
    elif HAS_BOTH_WEIGHTS:
        b_x1_hat = b_x1_hat * b_w1
        b_x2_hat = b_x2_hat * b_w2
        if HAS_BIAS1:
            b_x1_hat = b_x1_hat + b_b1
        if HAS_BIAS2:
            b_x2_hat = b_x2_hat + b_b2
        if HAS_X_WEIGHTS:
            b_y = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w
    else:
        # we set by default that if weight1 does not exist, weights2 will be ignored
        if HAS_BIAS1:
            b_x1_hat = b_x1_hat + b_b1
        if HAS_BIAS2:
            b_x2_hat = b_x2_hat + b_b2
        b_y = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w if HAS_X_WEIGHTS else b_x1_hat + b_x2_hat

    # swish/sigmoid output gate
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == 'swish':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


@triton.heuristics({
    'HAS_X_WEIGHTS': lambda args: args['x_weights_after_norm'] is not None,
    'HAS_DRESIDUAL': lambda args: args['dresidual1'] is not None,
    'HAS_WEIGHT1': lambda args: args['w1'] is not None,
    'HAS_WEIGHT2': lambda args: args['w2'] is not None,
    'HAS_BIAS1': lambda args: args['b1'] is not None,
    'HAS_BIAS2': lambda args: args['b2'] is not None,
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=['D', 'NB', 'IS_RMS_NORM', 'HAS_DRESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs
)
@triton.jit
def multi_input_layer_norm_gated_bwd_kernel(
        x1,  # pointer to the first input
        x2,  # pointer to the second input
        g,  # pointer to the gate
        x_weights_after_norm, # pointer to the x weights after norm
        w1,  # pointer to the weights
        b1,  # pointer to the biases
        w2,  # pointer to the weights
        b2,  # pointer to the biases
        y,  # pointer to the output to be recomputed
        dy,  # pointer to the output gradient
        dx1,  # pointer to the first input gradient
        dx2,  # pointer to the second input gradient
        dg,  # pointer to the gate gradient
        d_xw,  # pointer to the weights for x after normalization
        dw1,  # pointer to the partial sum of weights gradient
        dw2, # pointer to the weights for x2 after normalization
        db1,  # pointer to the partial sum of biases gradient
        db2, # pointer to the partial sum of biases gradient
        dresidual1,
        dresidual2,
        dresidual_in,
        mean1,
        rstd1,
        mean2,
        rstd2,
        T,
        BS,
        D: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        NB: tl.constexpr,
        ACTIVATION: tl.constexpr,
        IS_RMS_NORM: tl.constexpr,
        STORE_DRESIDUAL: tl.constexpr,
        HAS_X_WEIGHTS: tl.constexpr,
        HAS_DRESIDUAL: tl.constexpr,
        HAS_WEIGHT1: tl.constexpr,
        HAS_BIAS1: tl.constexpr,
        HAS_WEIGHT2: tl.constexpr,
        HAS_BIAS2: tl.constexpr,
        RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if HAS_WEIGHT1:
        b_w1 = tl.load(w1 + o_d, mask=m_d).to(tl.float32)
        b_dw1 = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_WEIGHT2:
        b_w2 = tl.load(w2 + o_d, mask=m_d).to(tl.float32)
        b_dw2 = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS1:
        b_b1 = tl.load(b1 + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_db1 = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS2:
        b_b2 = tl.load(b2 + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_db2 = tl.zeros((BT, BD), dtype=tl.float32)

    T = min(i_s * BS + BS, T)

    HAS_ONLY_WEIGHT1 = HAS_WEIGHT1 and not HAS_WEIGHT2
    HAS_BOTH_WEIGHTS = HAS_WEIGHT1 and HAS_WEIGHT2

    for i_t in range(i_s * BS, T, BT):
        p_x1 = tl.make_block_ptr(x1, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_x2 = tl.make_block_ptr(x2, (T, D), (D, 1), (i_t, 0), (BT, BD), (1,0))

        p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dx1 = tl.make_block_ptr(dx1, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dx2 = tl.make_block_ptr(dx2, (T, D), (D, 1), (i_t, 0), (BT, BD), (1,0))
        p_dg = tl.make_block_ptr(dg, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        # [BT, BD]
        b_x1 = tl.load(p_x1, boundary_check=(0, 1)).to(tl.float32)
        b_x2 = tl.load(p_x2, boundary_check=(0,1)).to(tl.float32)

        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)

        if HAS_X_WEIGHTS:
            p_x1_wn = tl.make_block_ptr(x_weights_after_norm, (T, 2), (2, 1), (i_t, 0), (BT, 1), (1,0))
            p_x2_wn = tl.make_block_ptr(x_weights_after_norm, (T, 2), (2, 1), (i_t, 1), (BT, 1), (1,0))

            b_x1_w = tl.load(p_x1_wn, boundary_check=(0,1)).to(tl.float32)
            b_x2_w = tl.load(p_x2_wn, boundary_check=(0,1)).to(tl.float32)

        if not IS_RMS_NORM:
            p_mean1 = tl.make_block_ptr(mean1, (T,), (1,), (i_t,), (BT,), (0,))
            b_mean1 = tl.load(p_mean1, boundary_check=(0,))

            p_mean2 = tl.make_block_ptr(mean2, (T,), (1,), (i_t,), (BT,), (0,))
            b_mean2 = tl.load(p_mean2, boundary_check=(0,))

        p_rstd1 = tl.make_block_ptr(rstd1, (T,), (1,), (i_t,), (BT,), (0,))
        b_rstd1 = tl.load(p_rstd1, boundary_check=(0,))
        # Compute dx
        b_x1_hat = (b_x1 - b_mean1[:, None]) * b_rstd1[:, None] if not IS_RMS_NORM else b_x1 * b_rstd1[:, None]
        b_x1_hat = tl.where(m_d[None, :], b_x1_hat, 0.0)

        p_rstd2 = tl.make_block_ptr(rstd2, (T,), (1,), (i_t,), (BT,), (0,))
        b_rstd2 = tl.load(p_rstd2, boundary_check=(0,))
        # Compute dx
        b_x2_hat = (b_x2 - b_mean2[:, None]) * b_rstd2[:, None] if not IS_RMS_NORM else b_x2 * b_rstd2[:, None]
        b_x2_hat = tl.where(m_d[None, :], b_x2_hat, 0.0)
        """
        if HAS_X_WEIGHTS:
            b_xhat = b_xhat1 * b_x1_weights_after_norm + b_xhat2 * b_x2_weights_after_norm
        else:
            b_xhat = b_xhat1 + b_xhat2

        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]
        if RECOMPUTE_OUTPUT:
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
        """
        if HAS_ONLY_WEIGHT1:
            b_x1_hat_ = b_x1_hat
            b_x2_hat_ = b_x2_hat
            if HAS_X_WEIGHTS:
                b_x_hat = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w
            else:
                b_x_hat = (b_x1_hat + b_x2_hat)
            b_y = b_x_hat * b_w1[None, :] if HAS_WEIGHT1 else b_x_hat
            if HAS_BIAS1:
                b_y = b_y + b_b1[None, :]
        elif HAS_BOTH_WEIGHTS:
            b_x1_hat_ = b_x1_hat * b_w1[None, :]
            b_x2_hat_ = b_x2_hat * b_w2[None, :]
            if HAS_BIAS1:
                b_x1_hat_ = b_x1_hat_ + b_b1[None, :]
            if HAS_BIAS2:
                b_x2_hat_ = b_x2_hat_ + b_b2[None, :]
            if HAS_X_WEIGHTS:
                b_y = b_x1_hat_ * b_x1_w + b_x2_hat_ * b_x2_w
            else:
                b_y = b_x1_hat_ + b_x2_hat_
        else:
            b_x1_hat_ = b_x1_hat
            b_x2_hat_ = b_x2_hat
            # we set by default that if weight1 does not exist, weights2 will be ignored
            if HAS_BIAS1:
                b_x1_hat = b_x1_hat + b_b1[None, :]
            if HAS_BIAS2:
                b_x2_hat = b_x2_hat + b_b2[None, :]
            b_y = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w if HAS_X_WEIGHTS else b_x1_hat + b_x2_hat

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 'swish':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'silu':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'sigmoid':
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy

        m_t = (i_t + tl.arange(0, BT)) < T
        if HAS_BOTH_WEIGHTS:
            # In this case, we have b_x_hat = (b_x1_hat * w1 * b_wx1 + b_x2_hat * w2 * b_wx2) *
            if HAS_X_WEIGHTS:
                b_dy1 = b_wdy * b_x1_w
                b_dy2 = b_wdy * b_x2_w

                b_dxw1 = tl.sum(b_wdy * b_x1_hat_, axis=1, keep_dims=True)
                b_dxw2 = tl.sum(b_wdy * b_x2_hat_, axis=1, keep_dims=True)

                p_dx1_wn = tl.make_block_ptr(d_xw, (T, 2), (2, 1), (i_t, 0), (BT, 1), (1, 0))
                p_dx2_wn = tl.make_block_ptr(d_xw, (T, 2), (2, 1), (i_t, 1), (BT, 1), (1, 0))
                tl.store(p_dx1_wn, b_dxw1.to(p_dx1_wn.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dx2_wn, b_dxw2.to(p_dx2_wn.dtype.element_ty), boundary_check=(0, 1))

                if HAS_BIAS1:
                    b_db1 += tl.where(m_t[:, None], b_dy1, 0.0)
                    b_db2 += tl.where(m_t[:, None], b_dy2, 0.0)

                b_wdy1 = b_dy1 * b_w1
                b_wdy2 = b_dy2 * b_w2
                if HAS_WEIGHT1:
                    b_dw1 += tl.where(m_t[:, None], b_dy1 * b_x1_hat, 0.0)
                    b_dw2 += tl.where(m_t[:, None], b_dy2 * b_x2_hat, 0.0)
            else:
                b_wdy1 = b_wdy * b_w1
                b_wdy2 = b_wdy * b_w2
                if HAS_BIAS1:
                    b_db1 += tl.where(m_t[:, None], b_dy, 0.0)
                    b_db2 += tl.where(m_t[:, None], b_dy, 0.0)

                if HAS_WEIGHT1:
                    b_dw1 += tl.where(m_t[:, None], b_dy * b_x1_hat, 0.0)
                    b_dw2 += tl.where(m_t[:, None], b_dy * b_x2_hat, 0.0)

            if not IS_RMS_NORM:
                b_c11 = tl.sum(b_x1_hat * b_wdy1, axis=1) / D
                b_c21 = tl.sum(b_wdy1, axis=1) / D
                b_dx1 = (b_wdy1 - (b_x1_hat * b_c11[:, None] + b_c21[:, None])) * b_rstd1[:, None]

                b_c12 = tl.sum(b_x2_hat * b_wdy2, axis=1) / D
                b_c22 = tl.sum(b_wdy2, axis=1) / D
                b_dx2 = (b_wdy2 - (b_x2_hat * b_c12[:, None] + b_c22[:, None])) * b_rstd2[:, None]
            else:
                b_c11 = tl.sum(b_x1_hat * b_wdy1, axis=1) / D
                b_dx1 = (b_wdy1 - b_x1_hat * b_c11[:, None]) * b_rstd1[:, None]

                b_c12 = tl.sum(b_x2_hat * b_wdy2, axis=1) / D
                b_dx2 = (b_wdy2 - b_x2_hat * b_c12[:, None]) * b_rstd2[:, None]
        else:
            if HAS_WEIGHT1:
                b_wdy = b_dy * b_w1
                b_dw1 += tl.where(m_t[:, None], b_dy * b_x1_hat, 0.0)
            if HAS_BIAS1:
                b_db1 += tl.where(m_t[:, None], b_dy, 0.0)

            if not IS_RMS_NORM:
                b_c11 = tl.sum(b_x1_hat * b_wdy, axis=1) / D
                b_c12 = tl.sum(b_x2_hat * b_wdy, axis=1) / D
                b_c2 = tl.sum(b_wdy, axis=1) / D
                if HAS_X_WEIGHTS:
                    b_dx1 = b_x1_w * (b_wdy - (b_x1_hat * b_c11[:, None] + b_c2[:, None])) * b_rstd1[:, None]
                    b_dx2 = b_x2_w * (b_wdy - (b_x2_hat * b_c12[:, None] + b_c2[:, None])) * b_rstd2[:, None]

                    b_dxw1 = tl.sum(b_wdy * b_x1_hat, axis=1)
                    b_dxw2 = tl.sum(b_wdy * b_x2_hat, axis=1)

                    p_dx1_wn = tl.make_block_ptr(d_xw, (T, 2), (2, 1), (i_t, 0), (BT, 1), (1, 0))
                    p_dx2_wn = tl.make_block_ptr(d_xw, (T, 2), (2, 1), (i_t, 1), (BT, 1), (1, 0))
                    tl.store(p_dx1_wn, b_dxw1.to(p_dx1_wn.dtype.element_ty), boundary_check=(0, 1))
                    tl.store(p_dx2_wn, b_dxw2.to(p_dx2_wn.dtype.element_ty), boundary_check=(0, 1))
                else:
                    b_dx1 = (b_wdy - (b_x1_hat * b_c11[:, None] + b_c2[:, None])) * b_rstd1[:, None]
                    b_dx2 = (b_wdy - (b_x2_hat * b_c12[:, None] + b_c2[:, None])) * b_rstd2[:, None]

            else:
                b_c11 = tl.sum(b_x1_hat * b_wdy, axis=1) / D
                b_c12 = tl.sum(b_x2_hat * b_wdy, axis=1) / D
                if HAS_X_WEIGHTS:
                    b_dx1 = b_x1_w * (b_wdy - b_x1_hat * b_c11[:, None]) * b_rstd1[:, None]
                    b_dx2 = b_x2_w * (b_wdy - b_x2_hat * b_c12[:, None]) * b_rstd2[:, None]

                    b_dxw1 = tl.sum(b_wdy * b_x1_hat, axis=1, keep_dims=True)
                    b_dxw2 = tl.sum(b_wdy * b_x2_hat, axis=1, keep_dims=True)

                    p_dx1_wn = tl.make_block_ptr(d_xw, (T, 2), (2, 1), (i_t, 0), (BT, 1), (1, 0))
                    p_dx2_wn = tl.make_block_ptr(d_xw, (T, 2), (2, 1), (i_t, 1), (BT, 1), (1, 0))
                    tl.store(p_dx1_wn, b_dxw1.to(p_dx1_wn.dtype.element_ty), boundary_check=(0, 1))
                    tl.store(p_dx2_wn, b_dxw2.to(p_dx2_wn.dtype.element_ty), boundary_check=(0, 1))

                else:
                    b_dx1 = (b_wdy - b_x1_hat * b_c11[:, None]) * b_rstd1[:, None]
                    b_dx2 = (b_wdy - b_x2_hat * b_c12[:, None]) * b_rstd2[:, None]



        if HAS_DRESIDUAL:
            p_dres1 = tl.make_block_ptr(dresidual1, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            b_dres1 = tl.load(p_dres1, boundary_check=(0, 1)).to(tl.float32)
            b_dx1 += b_dres1

            p_dres2 = tl.make_block_ptr(dresidual2, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            b_dres2 = tl.load(p_dres2, boundary_check=(0, 1)).to(tl.float32)
            b_dx2 += b_dres2

        # Write dx
        if STORE_DRESIDUAL:
            p_dres_in = tl.make_block_ptr(dresidual_in, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_dres_in, (b_dx2 + b_dx1).to(p_dres_in.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_dx1, b_dx1.to(p_dx1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dx2, b_dx2.to(p_dx1.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    if HAS_WEIGHT1:
        tl.store(dw1 + i_s * D + o_d, tl.sum(b_dw1, axis=0), mask=m_d)
    if HAS_BIAS1:
        tl.store(db1 + i_s * D + o_d, tl.sum(b_db1, axis=0), mask=m_d)
    if HAS_WEIGHT2:
        tl.store(dw2 + i_s * D + o_d, tl.sum(b_dw2, axis=0), mask=m_d)
    if HAS_BIAS2:
        tl.store(db2 + i_s * D + o_d, tl.sum(b_db2, axis=0), mask=m_d)


@triton.heuristics({
    'HAS_X_WEIGHTS': lambda args: args['x_weights_after_norm'] is not None,
    'HAS_DRESIDUAL': lambda args: args['dresidual'] is not None,
    'HAS_WEIGHT1': lambda args: args['w1'] is not None,
    'HAS_WEIGHT2': lambda args: args['w2'] is not None,
    'HAS_BIAS1': lambda args: args['b1'] is not None,
    'HAS_BIAS2': lambda args: args['b2'] is not None,
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['D', 'IS_RMS_NORM', 'STORE_DRESIDUAL', 'HAS_DRESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs
)
@triton.jit
def multi_input_layer_norm_gated_bwd_kernel1(
        x1,  # pointer to the first input
        x2,  # pointer to the second input
        g,  # pointer to the gate
        x_weights_after_norm,  # pointer to the x weights after norm
        w1,  # pointer to the weights
        b1,  # pointer to the biases
        w2,  # pointer to the weights
        b2,  # pointer to the biases
        y,  # pointer to the output to be recomputed
        dy,  # pointer to the output gradient
        dx1,  # pointer to the first input gradient
        dx2,  # pointer to the second input gradient
        dg,  # pointer to the gate gradient
        d_xw,  # pointer to the weights for x after normalization
        dw1,  # pointer to the partial sum of weights gradient
        dw2, # pointer to the weights for x2 after normalization
        db1,  # pointer to the partial sum of biases gradient
        db2, # pointer to the partial sum of biases gradient
        dresidual1,
        dresidual2,
        dresidual_in,
        mean1,
        rstd1,
        mean2,
        rstd2,
        T,
        BS,
        D: tl.constexpr,
        BD: tl.constexpr,
        ACTIVATION: tl.constexpr,
        IS_RMS_NORM: tl.constexpr,
        STORE_DRESIDUAL: tl.constexpr,
        HAS_X_WEIGHTS: tl.constexpr,
        HAS_DRESIDUAL: tl.constexpr,
        HAS_WEIGHT1: tl.constexpr,
        HAS_BIAS1: tl.constexpr,
        HAS_WEIGHT2: tl.constexpr,
        HAS_BIAS2: tl.constexpr,
        RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    mask = o_d < D
    x1 += i_s * BS * D
    x2 += i_s * BS * D
    g += i_s * BS * D
    if HAS_DRESIDUAL:
        dresidual1 += i_s * BS * D
        dresidual2 += i_s * BS * D
    if STORE_DRESIDUAL:
        dresidual_in += i_s * BS * D

    dy += i_s * BS * D
    dx1 += i_s * BS * D
    dx2 += i_s * BS *D
    dg += i_s * BS * D
    if RECOMPUTE_OUTPUT:
        y += i_s * BS * D
    if HAS_X_WEIGHTS:
        x_weights_after_norm += i_s * BS * 2
        d_xw += i_s * BS * 2
    if HAS_WEIGHT1:
        b_w1 = tl.load(w1 + o_d, mask=mask).to(tl.float32)
        b_dw1 = tl.zeros((BD,), dtype=tl.float32)
    if HAS_WEIGHT2:
        b_w2 = tl.load(w2 + o_d, mask=mask).to(tl.float32)
        b_dw2 = tl.zeros((BD,), dtype=tl.float32)

    if HAS_BIAS1:
        b_b1 = tl.load(b1 + o_d, mask=mask, other=0.0).to(tl.float32)
        b_db1 = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS2:
        b_b2 = tl.load(b2 + o_d, mask=mask, other=0.0).to(tl.float32)
        b_db2 = tl.zeros((BD,), dtype=tl.float32)

    HAS_ONLY_WEIGHT1 = HAS_WEIGHT1 and not HAS_WEIGHT2
    HAS_BOTH_WEIGHTS = HAS_WEIGHT1 and HAS_WEIGHT2

    for i_t in range(i_s * BS, min(i_s * BS + BS, T)):
        # Load data to SRAM
        b_x1 = tl.load(x1 + o_d, mask=mask, other=0).to(tl.float32)
        b_x2 = tl.load(x2 + o_d, mask=mask, other=0).to(tl.float32)

        b_g = tl.load(g + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + o_d, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            b_mean1 = tl.load(mean1 + i_t)
            b_mean2 = tl.load(mean2 + i_t)
        b_rstd1 = tl.load(rstd1 + i_t)
        b_rstd2 = tl.load(rstd2 + i_t)

        # Compute dx
        b_xhat1 = (b_x1 - b_mean1) * b_rstd1 if not IS_RMS_NORM else b_x1 * b_rstd1
        b_xhat1 = tl.where(mask, b_xhat1, 0.0)

        b_xhat2 = (b_x2 - b_mean2) * b_rstd2 if not IS_RMS_NORM else b_x2 * b_rstd2
        b_xhat2 = tl.where(mask, b_xhat2, 0.0)

        if HAS_X_WEIGHTS:
            b_x1_w = tl.load(x_weights_after_norm + i_t * 2).to(tl.float32)
            b_x2_w = tl.load(x_weights_after_norm + i_t * 2 + 1).to(tl.float32)

        if HAS_ONLY_WEIGHT1:
            b_x1_hat_ = b_x1_hat
            b_x2_hat_ = b_x2_hat
            if HAS_X_WEIGHTS:
                b_x_hat = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w
            else:
                b_x_hat = (b_x1_hat + b_x2_hat)
            b_y = b_x_hat * b_w1[None, :] if HAS_WEIGHT1 else b_x_hat
            if HAS_BIAS1:
                b_y = b_y + b_b1[None, :]
        elif HAS_BOTH_WEIGHTS:
            b_x1_hat_ = b_x1_hat * b_w1[None, :]
            b_x2_hat_ = b_x2_hat * b_w2[None, :]
            if HAS_BIAS1:
                b_x1_hat_ = b_x1_hat_ + b_b1[None, :]
            if HAS_BIAS2:
                b_x2_hat_ = b_x2_hat_ + b_b2[None, :]
            if HAS_X_WEIGHTS:
                b_y = b_x1_hat_ * b_x1_w + b_x2_hat_ * b_x2_w
            else:
                b_y = b_x1_hat_ + b_x2_hat_
        else:
            b_x1_hat_ = b_x1_hat
            b_x2_hat_ = b_x2_hat
            # we set by default that if weight1 does not exist, weights2 will be ignored
            if HAS_BIAS1:
                b_x1_hat = b_x1_hat + b_b1[None, :]
            if HAS_BIAS2:
                b_x2_hat = b_x2_hat + b_b2[None, :]
            b_y = b_x1_hat * b_x1_w + b_x2_hat * b_x2_w if HAS_X_WEIGHTS else b_x1_hat + b_x2_hat

        if RECOMPUTE_OUTPUT:
            tl.store(y + o_d, b_y, mask=mask)

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 'swish':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'silu':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'sigmoid':
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g


        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += b_dy * b_xhat
        if HAS_BIAS:
            b_db += b_dy

        if not IS_RMS_NORM:
            b_c11 = tl.sum(b_xhat1 * b_wdy, axis=0) / D
            b_c12 = tl.sum(b_xhat2 * b_wdy, axis=0) / D
            b_c2 = tl.sum(b_wdy, axis=0) / D
            if HAS_X_WEIGHTS:
                b_dx1 = (b_wdy - (b_xhat1 * b_c11 + b_c2)) * b_rstd1 * b_x1_weights_after_norm
                b_dx2 = (b_wdy - (b_xhat2 * b_c12 + b_c2)) * b_rstd2 * b_x2_weights_after_norm

                b_dxw1 = tl.sum(b_wdy * b_xhat1, )
                b_dxw2 = tl.sum(b_wdy * b_xhat2, )

                tl.store(d_xw + i_t * 2, b_dxw1.to(d_xw.dtype.element_ty))
                tl.store(d_xw + i_t * 2 + 1, b_dxw2.to(d_xw.dtype.element_ty))
            else:
                b_dx1 = (b_wdy - (b_xhat1 * b_c11 + b_c2)) * b_rstd1
                b_dx2 = (b_wdy - (b_xhat2 * b_c12 + b_c2)) * b_rstd2

        else:
            b_c11 = tl.sum(b_xhat1 * b_wdy, axis=0) / D
            b_c12 = tl.sum(b_xhat2 * b_wdy, axis=0) / D
            if HAS_X_WEIGHTS:
                b_dx1 = (b_wdy - b_xhat1 * b_c11) * b_rstd1 * b_x1_weights_after_norm
                b_dx2 = (b_wdy - b_xhat2 * b_c12) * b_rstd2 * b_x2_weights_after_norm

                b_dxw1 = tl.sum(b_wdy * b_xhat1, )
                b_dxw2 = tl.sum(b_wdy * b_xhat2, )

                tl.store(d_xw + i_t * 2, b_dxw1.to(d_xw.dtype.element_ty))
                tl.store(d_xw + i_t * 2 + 1, b_dxw2.to(d_xw.dtype.element_ty))
            else:
                b_dx1 = (b_wdy - b_xhat1 * b_c11) * b_rstd1
                b_dx2 = (b_wdy - b_xhat2 * b_c12) * b_rstd2

        if HAS_BOTH_WEIGHTS:
            # In this case, we have b_x_hat = (b_x1_hat * w1 * b_wx1 + b_x2_hat * w2 * b_wx2) *
            if HAS_X_WEIGHTS:
                b_dy1 = b_wdy * b_x1_w
                b_dy2 = b_wdy * b_x2_w

                b_dxw1 = tl.sum(b_wdy * b_x1_hat_, )
                b_dxw2 = tl.sum(b_wdy * b_x2_hat_, )
                tl.store(d_xw + i_t * 2, b_dxw1.to(d_xw.dtype.element_ty))
                tl.store(d_xw + i_t * 2 + 1, b_dxw2.to(d_xw.dtype.element_ty))

                if HAS_BIAS1:
                    b_db1 += b_dy1
                    b_db2 += b_dy2

                b_wdy1 = b_dy1 * b_w1
                b_wdy2 = b_dy2 * b_w2
                if HAS_WEIGHT1:
                    b_dw1 += b_dy1 * b_x1_hat
                    b_dw2 += b_dy2 * b_x2_hat
            else:
                b_wdy1 = b_wdy * b_w1
                b_wdy2 = b_wdy * b_w2
                if HAS_BIAS1:
                    b_db1 += b_dy
                    b_db2 += b_dy

                if HAS_WEIGHT1:
                    b_dw1 += b_dy * b_x1_hat
                    b_dw2 += b_dy * b_x2_hat

            if not IS_RMS_NORM:
                b_c11 = tl.sum(b_x1_hat * b_wdy1, axis=0) / D
                b_c21 = tl.sum(b_wdy1, axis=0) / D
                b_dx1 = (b_wdy1 - (b_x1_hat * b_c11 + b_c21)) * b_rstd1

                b_c12 = tl.sum(b_x2_hat * b_wdy2, axis=0) / D
                b_c22 = tl.sum(b_wdy2, axis=0) / D
                b_dx2 = (b_wdy2 - (b_x2_hat * b_c12 + b_c22)) * b_rstd2
            else:
                b_c11 = tl.sum(b_x1_hat * b_wdy1, axis=0) / D
                b_dx1 = (b_wdy1 - b_x1_hat * b_c11) * b_rstd1

                b_c12 = tl.sum(b_x2_hat * b_wdy2, axis=0) / D
                b_dx2 = (b_wdy2 - b_x2_hat * b_c12) * b_rstd2
        else:
            if HAS_WEIGHT1:
                b_wdy = b_dy * b_w1
                b_dw1 +=  b_dy * b_x1_hat
            if HAS_BIAS1:
                b_db1 += b_dy

            if not IS_RMS_NORM:
                b_c11 = tl.sum(b_x1_hat * b_wdy, axis=0) / D
                b_c12 = tl.sum(b_x2_hat * b_wdy, axis=0) / D
                b_c2 = tl.sum(b_wdy, axis=0) / D
                if HAS_X_WEIGHTS:
                    b_dx1 = b_x1_w * (b_wdy - (b_x1_hat * b_c11 + b_c2)) * b_rstd1
                    b_dx2 = b_x2_w * (b_wdy - (b_x2_hat * b_c12 + b_c2)) * b_rstd2

                    b_dxw1 = tl.sum(b_wdy * b_x1_hat, axis=1)
                    b_dxw2 = tl.sum(b_wdy * b_x2_hat, axis=1)

                    tl.store(d_xw + i_t * 2, b_dxw1.to(d_xw.dtype.element_ty))
                    tl.store(d_xw + i_t * 2 + 1, b_dxw2.to(d_xw.dtype.element_ty))
                else:
                    b_dx1 = (b_wdy - (b_x1_hat * b_c11 + b_c2)) * b_rstd1
                    b_dx2 = (b_wdy - (b_x2_hat * b_c12 + b_c2)) * b_rstd2

            else:
                b_c11 = tl.sum(b_x1_hat * b_wdy, axis=0) / D
                b_c12 = tl.sum(b_x2_hat * b_wdy, axis=0) / D
                if HAS_X_WEIGHTS:
                    b_dx1 = b_x1_w * (b_wdy - b_x1_hat * b_c11) * b_rstd1
                    b_dx2 = b_x2_w * (b_wdy - b_x2_hat * b_c12) * b_rstd2

                    b_dxw1 = tl.sum(b_wdy * b_x1_hat,  )
                    b_dxw2 = tl.sum(b_wdy * b_x2_hat, )

                    tl.store(d_xw + i_t * 2, b_dxw1.to(d_xw.dtype.element_ty))
                    tl.store(d_xw + i_t * 2 + 1, b_dxw2.to(d_xw.dtype.element_ty))
                else:
                    b_dx1 = (b_wdy - b_x1_hat * b_c11[:, None]) * b_rstd1[:, None]
                    b_dx2 = (b_wdy - b_x2_hat * b_c12[:, None]) * b_rstd2[:, None]

        if HAS_DRESIDUAL:
            b_dres1 = tl.load(dresidual1 + o_d, mask=mask, other=0).to(tl.float32)
            b_dx1 += b_dres1

            b_dres2 = tl.load(dresidual2 + o_d, mask=mask, other=0).to(tl.float32)
            b_dx2 += b_dres2

        # Write dx
        if STORE_DRESIDUAL:
            tl.store(dresidual_in + o_d, b_dx1 + b_dx2, mask=mask)

        tl.store(dx1 + o_d, b_dx1, mask=mask)
        tl.store(dg + o_d, b_dg, mask=mask)

        x1 += D
        x2 += D
        g += D
        if HAS_DRESIDUAL:
            dresidual1 += D
            dresidual2 += D
        if STORE_DRESIDUAL:
            dresidual_in += D
        if RECOMPUTE_OUTPUT:
            y += D
        dy += D
        dx1 += D
        dx2 += D
        dg += D
    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)


def multi_input_layer_norm_gated_fwd(
        x: tuple[torch.Tensor],
        g: torch.Tensor,
        x_weights_after_norm: Optional[torch.Tensor],
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        bias1: torch.Tensor,
        bias2:torch.Tensor,
        activation: str = 'swish',
        eps: float = 1e-5,
        residual: torch.Tensor = None,
        out_dtype: torch.dtype = None,
        residual_dtype: torch.dtype = None,
        is_rms_norm: bool = False
):
    if residual is not None:
        residual_dtype = residual.dtype
    x0 = x[0]
    T, D = x0.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight1 is not None:
        assert weight1.shape == (D,)
    if weight2 is not None:
        assert weight2.shape == (D,)
    if bias1 is not None:
        assert bias1.shape == (D,)
    if bias2 is not None:
        assert bias2.shape == (D,)
    # allocate output
    y = torch.empty_like(x0, dtype=x0.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x0.dtype):
        residual_out = tuple(torch.empty(T, D, device=x0.device, dtype=residual_dtype) for _ in range(len(x)))
    else:
        residual_out = None
    mean = tuple(torch.empty((T,), dtype=torch.float, device=x0.device) for _ in range(len(x))) if not is_rms_norm else None
    rstd = tuple(torch.empty((T,), dtype=torch.float, device=x0.device) for _ in range(len(x)))
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x0.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta['BT']),)

        multi_input_layer_norm_gated_fwd_kernel[grid](
            x1=x[0],
            x2=x[1],
            g=g,
            y=y,
            x_weights_after_norm=x_weights_after_norm,
            w1=weight1,
            b1=bias1,
            w2=weight2,
            b2=bias2,
            residual=residual,
            residual_out1=residual_out[0] if residual_out is not None else None,
            residual_out2=residual_out[1] if residual_out is not None else None,
            mean1=mean[0] if mean is not None else None,
            mean2=mean[1] if mean is not None else None,
            rstd1=rstd[0],
            rstd2=rstd[1],
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
        )
    else:
        multi_input_layer_norm_gated_fwd_kernel1[(T,)](
            x1=x[0],
            x2=x[1],
            g=g,
            y=y,
            x_weights_after_norm=x_weights_after_norm,
            w1=weight1,
            b1=bias1,
            w2=weight2,
            b2=bias2,
            residual=residual,
            residual_out1=residual_out[0],
            residual_out2=residual_out[1],
            mean1=mean[0],
            mean2=mean[1],
            rstd1=rstd[0],
            rstd2=rstd[1],
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


def multi_input_layer_norm_gated_bwd(
        dy: torch.Tensor,
        x: tuple[torch.Tensor],
        g: torch.Tensor,
        x_weights_after_norm: Optional[torch.Tensor],
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        bias1: torch.Tensor,
        bias2: torch.Tensor,
        activation: str = 'swish',
        eps: float = 1e-5,
        mean: tuple[torch.Tensor] = None,
        rstd: tuple[torch.Tensor] = None,
        dresidual: tuple[torch.Tensor] = None,
        has_residual: bool = False,
        is_rms_norm: bool = False,
        x_dtype: torch.dtype = None,
        recompute_output: bool = False,
):
    x0 = x[0]
    T, D = x0.shape
    assert dy.shape == (T, D)
    if dresidual is not None:
        assert dresidual[0].shape == (T, D)
    if weight1 is not None:
        assert weight1.shape == (D,)
    if weight2 is not None:
        assert weight2.shape == (D,)
    if bias1 is not None:
        assert bias1.shape == (D,)
    if bias2 is not None:
        assert bias2.shape == (D,)
    # allocate output
    dx = tuple(torch.empty_like(x0) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x0.device) for _ in range(len(x)))
    dg = torch.empty_like(g) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x0.device)
    dresidual_in = torch.empty_like(x0) if has_residual and dx[0].dtype != x[0].dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    if x_weights_after_norm is not None:
        d_xw = torch.empty_like(x_weights_after_norm)
    else:
        d_xw = None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x0.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    NS = get_multiprocessor_count(x0.device.index)
    BS = math.ceil(T / NS)

    dw1 = torch.empty((NS, D), dtype=torch.float, device=weight1.device) if weight1 is not None else None
    dw2 = torch.empty((NS, D), dtype=torch.float, device=weight2.device) if weight2 is not None else None
    db1 = torch.empty((NS, D), dtype=torch.float, device=bias1.device) if bias1 is not None else None
    db2 = torch.empty((NS, D), dtype=torch.float, device=bias2.device) if bias2 is not None else None
    grid = (NS,)

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        multi_input_layer_norm_gated_bwd_kernel[grid](
            x1=x[0],
            x2=x[1],
            g=g,
            x_weights_after_norm=x_weights_after_norm,
            w1=weight1,
            w2=weight2,
            b1=bias1,
            b2=bias2,
            y=y,
            dy=dy,
            dx1=dx[0],
            dx2=dx[1],
            dg=dg,
            d_xw=d_xw,
            dw1=dw1,
            dw2=dw2,
            db1=db1,
            db2=db2,
            dresidual1=dresidual[0] if dresidual is not None else None,
            dresidual2=dresidual[1] if dresidual is not None else None,
            dresidual_in=dresidual_in,
            mean1=mean[0] if mean is not None else None,
            mean2=mean[1] if mean is not None else None,
            rstd1=rstd[0] if rstd is not None else None,
            rstd2=rstd[1] if rstd is not None else None,
            T=T,
            D=D,
            BS=BS,
            BD=BD,
            NB=NB,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_DRESIDUAL=dresidual_in is not None,
        )
    else:
        multi_input_layer_norm_gated_bwd_kernel1[grid](
            x1=x[0],
            x2=x[1],
            g=g,
            x_weights_after_norm=x_weights_after_norm,
            w1=weight1,
            w2=weight2,
            b1=bias1,
            b2=bias2,
            y=y,
            dy=dy,
            dx1=dx[0],
            dx2=dx[1],
            dxw=d_xw,
            dw1=dw1,
            dw2=dw2,
            db1=db1,
            db2=db2,
            dresidual1=dresidual[0] if dresidual is not None else None,
            dresidual2=dresidual[1] if dresidual is not None else None,
            dresidual_in=dresidual_in,
            mean1=mean[0] if mean is not None else None,
            mean2=mean[1] if mean is not None else None,
            rstd1=rstd[0] if rstd is not None else None,
            rstd2=rstd[1] if rstd is not None else None,
            T=T,
            D=D,
            BS=BS,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_DRESIDUAL=dresidual_in is not None,
        )
    dw1 = dw1.sum(0).to(weight1.dtype) if weight1 is not None else None
    dw2 = dw2.sum(0).to(weight2.dtype) if weight2 is not None else None
    db1 = db1.sum(0).to(bias1.dtype) if bias1 is not None else None
    db2 = db2.sum(0).to(bias2.dtype) if bias2 is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dg, d_xw, dw1, dw2, db1, db2, dresidual_in) if not recompute_output else (dx, dg, d_xw, dw1, dw2, db1, db2, dresidual_in, y)


class TwoInputLayerNormGatedFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
            ctx,
            x1: torch.Tensor,
            x2: torch.Tensor,
            g: torch.Tensor,
            x_weights_after_norm: torch.Tensor,
            weight1: torch.Tensor,
            weight2: torch.Tensor,
            bias1: torch.Tensor,
            bias2: torch.Tensor,
            activation: str,
            residual: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
            prenorm: bool = False,
            residual_in_fp32: bool = False,
            is_rms_norm: bool = False,
    ):
        x_shape_og = x1.shape
        g_shape_og = g.shape
        # reshape input data into 2D tensor
        g = g.reshape(-1, g.shape[-1])
        if x_weights_after_norm is not None:
            assert x_weights_after_norm.shape[-1] == 2
            assert x_weights_after_norm.shape[-2] == x_shape_og[-2]
            x_weights_after_norm = x_weights_after_norm.reshape(-1, x_weights_after_norm.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = multi_input_layer_norm_gated_fwd(
            x=(x1.reshape(-1, x1.shape[-1]), x2.reshape(-1, x1.shape[-1])),
            x_weights_after_norm=x_weights_after_norm,
            g=g,
            weight1=weight1,
            weight2=weight2,
            bias1=bias1,
            bias2=bias2,
            activation=activation,
            eps=eps,
            residual=residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm
        )
        ctx.save_for_backward(residual_out[0], residual_out[1], g, x_weights_after_norm,
                              weight1, bias1, weight2, bias2,
                              mean[0] if mean is not None else None,
                              mean[1] if mean is not None else None,
                              rstd[0], rstd[1])
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.activation = activation
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x1.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out[0].reshape(x_shape_og), residual_out[1].reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dy, *args):
        x1, x2, g, x_weights_after_norm, weight1, bias1, weight2, bias2, mean1, mean2, rstd1, rstd2 = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == x1.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x1.shape
        else:
            dresidual = None
        dx, dg, dxw, dw1, db1, dw2, db2, dres_in = multi_input_layer_norm_gated_bwd(
            dy=dy,
            x=(x1,x2),
            g=g,
            x_weights_after_norm=x_weights_after_norm,
            weight1=weight1,
            weight2=weight2,
            bias1=bias1,
            bias2=bias2,
            activation=ctx.activation,
            eps=ctx.eps,
            mean=(mean1, mean2),
            rstd=(rstd1, rstd2),
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx[0].reshape(ctx.x_shape_og),
            dx[1].reshape(ctx.x_shape_og),
            dg.reshape(ctx.g_shape_og),
            dxw.reshape([*ctx.x_shape_og[:-1], 2]),
            dw1,
            db1,
            dw2,
            db2,
            None,
            dres_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def two_inputs_rms_norm_gated(
        x1: torch.Tensor,
        x2: torch.Tensor,
        g: torch.Tensor,
        x_weights_after_norm: torch.Tensor,
        weight1: torch.Tensor,
        weight2: Optional[torch.Tensor],
        bias1: torch.Tensor,
        bias2: Optional[torch.Tensor],
        activation: str = 'swish',
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        eps: float = 1e-6
):
    return TwoInputLayerNormGatedFunction.apply(
        x1,
        x2,
        g,
        x_weights_after_norm,
        weight1,
        weight2,
        bias1,
        bias2,
        activation,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True
    )


class FusedMultiInputRMSNormGated(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            elementwise_affine: bool = True,
            eps: float = 1e-5,
            activation: str = 'swish',
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> FusedMultiInputRMSNormGated:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ['swish', 'silu', 'sigmoid']:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight1 = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            self.weight2 = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight1", None)
            self.register_parameter("weight2", None)
        self.register_parameter("bias1", None)
        self.register_parameter("bias2", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight1)
            nn.init.ones_(self.weight2)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            g: torch.Tensor,
            x_weights_after_norm: Optional[torch.Tensor] = None,
            residual: Optional[torch.Tensor] = None,
            prenorm: bool = False,
            residual_in_fp32: bool = False
    ) -> torch.Tensor:
        return two_inputs_rms_norm_gated(
            x1,
            x2,
            g,
            x_weights_after_norm,
            self.weight1,
            self.weight2,
            self.bias1,
            self.bias2,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    torch.manual_seed(0)
    hs = 256
    norm = FusedMultiInputRMSNormGated(hs, device=torch.device('cuda'), eps=1e-5)

    x1 = torch.randn(4,512,hs).cuda()
    x2 = torch.randn(4,512,hs).cuda()
    xw = torch.randn(4,512, 2).cuda()

    d1 = nn.Parameter(copy.deepcopy(x1), requires_grad=True)
    d2 = nn.Parameter(copy.deepcopy(x2), requires_grad=True)
    xw1 = nn.Parameter(copy.deepcopy(xw), requires_grad=True)

    g = torch.randn(4,512,hs).sigmoid().cuda()

    g1 = nn.Parameter(copy.deepcopy(g), requires_grad=True)

    w1_data = torch.randn_like(norm.weight1.data)
    w2_data = torch.randn_like(norm.weight2.data)
    norm.weight1.data = w1_data.clone()
    norm.weight2.data = w2_data.clone()

    o1 = norm(d1, d2, g1, xw1)
    loss = o1.abs().sum()
    loss.backward()

    d11 = nn.Parameter(copy.deepcopy(x1), requires_grad=True)
    d21 = nn.Parameter(copy.deepcopy(x2), requires_grad=True)
    g11 = nn.Parameter(copy.deepcopy(g), requires_grad=True)
    xw2 = nn.Parameter(copy.deepcopy(xw), requires_grad=True)

    weight1 = nn.Parameter(w1_data.clone(), requires_grad=True)
    weight2 = nn.Parameter(w2_data.clone(), requires_grad=True)
    var1 = (d11*d11).sum(-1, keepdim=True)/hs
    vbar2 = (d21*d21).sum(-1, keepdim=True)/hs
    eps = 1e-5
    o2 = ((d11 /(torch.sqrt(var1) + eps)) * weight1[None, None, :] * xw2[..., [0]] + (d21 / (torch.sqrt(vbar2) + eps))* xw2[..., [1]]  * weight2[None, None, :]) * g11 * nn.functional.sigmoid(g11)
    loss2 = o2.abs().sum()
    loss2.backward()

    import pdb
    pdb.set_trace()

