# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External function interface to MPS libraries."""
import tvm
from tvm import te


# pylint: disable=C0103,W0612


def matmul(lhs, rhs, transa=False, transb=False):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS

    This function serves as an example on how to calle external libraries.

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    m = lhs.shape[0] if transa is False else lhs.shape[1]
    n = rhs.shape[1] if transb is False else rhs.shape[0]
    if transa:
        m = b
    if transb:
        n = c
    return te.extern(
        (m, n),
        [lhs, rhs],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mps.matmul", ins[0], ins[1], outs[0], transa, transb
        ),
        name="C",
    )


def conv2d(data, weight, pad="SAME", stride=1):
    """
    Create an extern op that compute data * weight and return result in output

    Parameters:
    ----------
    data: Tensor
        The input data, format NHWC
    weight: Tensor
        The conv weight, format output_feature * kH * kW * input_feature
    pad: str
        Padding method, 'SAME' or 'VALID'
    stride: int
        convolution stride

    Returns
    -------
    output: Tensor
        The result tensor
    """
    n, hi, wi, ci = data.shape
    co, kh, kw, ciw = weight.shape
    padding = 0 if pad == "SAME" else 1
    ho = hi // stride
    wo = wi // stride

    return te.extern(
        (n, ho, wo, co),
        [data, weight],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mps.conv2d", ins[0], ins[1], outs[0], padding, stride
        ),
        name="C",
    )


def lstm(
    input,
    weight_ih,
    weight_hh,
    bias_ih,
    bias_hh,
    h0,
    c0,
    hidden_size,
    num_layers=1,
    batch_first=False,
    bidirectional=False,
):
    """Create an extern op that runs an LSTM using MPS.

    Notes
    -----
    Current implementation limitations:
    - float32 only
    - num_layers=1 only
    - bidirectional=False only
    - h0/c0 inputs are accepted but may be ignored by the underlying runtime implementation.
    """
    if batch_first:
        batch, seq_len, _ = input.shape
    else:
        seq_len, batch, _ = input.shape

    return te.extern(
        [
            (batch, seq_len, hidden_size) if batch_first else (seq_len, batch, hidden_size),
            (1, batch, hidden_size),
            (1, batch, hidden_size),
        ],
        [input, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mps.lstm",
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            ins[4],
            ins[5],
            ins[6],
            outs[0],
            outs[1],
            outs[2],
            hidden_size,
            num_layers,
            batch_first,
            bidirectional,
        ),
        name="lstm_out",
    )


def lstm_packed(
    input,
    lengths,
    weight_ih,
    weight_hh,
    bias_ih,
    bias_hh,
    h0,
    c0,
    hidden_size,
    num_layers=1,
    batch_first=False,
    bidirectional=False,
    reverse=False,
):
    """Create an extern op that runs a length-aware (packed-semantics) LSTM using MPS.

    This variant carries `lengths` explicitly so a backend can implement true variable-length
    behavior (e.g. via MPS ragged-row encoding).
    """
    if batch_first:
        batch, seq_len, _ = input.shape
    else:
        seq_len, batch, _ = input.shape

    return te.extern(
        [
            (batch, seq_len, hidden_size) if batch_first else (seq_len, batch, hidden_size),
            (1, batch, hidden_size),
            (1, batch, hidden_size),
        ],
        [input, lengths, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mps.lstm_packed",
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            ins[4],
            ins[5],
            ins[6],
            ins[7],
            outs[0],
            outs[1],
            outs[2],
            hidden_size,
            num_layers,
            batch_first,
            bidirectional,
            reverse,
        ),
        dtype=["float32", "float32", "float32"],
        name="lstm_out_packed",
    )
