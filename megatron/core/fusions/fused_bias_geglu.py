# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.jit import jit_fuser










@jit_fuser
def geglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1)))) * y_2


@jit_fuser
def bias_geglu(bias, y):
    y = y + bias
    return geglu(y)





@jit_fuser
def geglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    tanh_out = torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))

    ff = 0.5 * y_1 * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y_1 * y_1)) + 0.5 * (
        1 + tanh_out
    )
    return torch.cat(((g * y_2) * ff, g * (y_1 * 0.5 * (1.0 + tanh_out))), -1)


@jit_fuser
def bias_geglu_back(g, y, bias):
    y = y + bias
    return geglu_back(g, y)


class BiasGeGLUFunction(torch.autograd.Function):
    @staticmethod

    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_geglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_geglu_back(grad_output, input, bias)
        return tmp, tmp


class GeGLUFunction(torch.autograd.Function):
    @staticmethod

    def forward(ctx, input):
        ctx.save_for_backward(input)
        return geglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = geglu_back(grad_output, input[0])
        return tmp


def bias_geglu_impl(input, bias):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasGeGLUFunction.apply(input, bias)
    else:
        output = GeGLUFunction.apply(input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
