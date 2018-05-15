import torch
import torch.nn as nn

import torch.nn.functional as F  # useful stateless functions
import math

#TODO: ADD more quantization function

class Quant:
    def linear(input, bits):
        assert bits >= 1, bits
        if bits == 1:
            return torch.sign(input) - 1
        sf = torch.ceil(torch.log2(torch.max(torch.abs(input))))
        delta = math.pow(2.0, -sf)
        bound = math.pow(2.0, bits-1)
        min_val = - bound
        max_val = bound - 1
        rounded = torch.floor(input / delta + 0.5)

        clipped_value = torch.clamp(rounded, min_val, max_val) * delta
        return clipped_value

class quantization(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  @staticmethod
  def forward(ctx, x, bits, quant_func):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
    """
    #Define a constant
    ctx.bits = bits
    #ctx.save_for_backward(x)
    clipped_value = quant_func(x, bits)
    return clipped_value

  def backward(ctx, grad_output):
    """
    In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
    """
    grad_x = grad_output.clone()
    return grad_x, None, None

class activation_quantization(nn.Module):
    def __init__(self, bits=8, quant_func=Quant.linear):
        super(activation_quantization, self).__init__()
        self.bits = bits
        self.func = quant_func

    def forward(self, inputActivation):
        return quantization.apply(inputActivation, self.bits, self.func)

