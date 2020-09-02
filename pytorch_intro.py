from __future__ import print_function
import torch

#Tensors

x = torch.empty(5, 3)
print(x)

"""
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
"""

x = torch.rand(5, 3)
print(x)

"""
tensor([[0.5426, 0.5621, 0.8710],
        [0.3132, 0.1949, 0.9939],
        [0.9807, 0.5656, 0.3241],
        [0.3242, 0.7866, 0.0351],
        [0.8095, 0.5571, 0.5852]])
"""

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

"""
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
"""

x = torch.tensor([5.5, 3])
print(x)

"""
tensor([5.5000, 3.0000])
"""

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

"""
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-0.0189,  1.1484,  1.0455],
        [ 1.9734,  1.1751,  0.2004],
        [-1.3734, -0.0671,  0.4612],
        [ 0.1066, -0.7407, -0.9855],
        [ 0.6659,  0.1239, -1.7233]])
"""

print(x.size())

"""
torch.Size([5, 3])
"""

# Operations

y = torch.rand(5, 3)
print(x + y)

"""
tensor([[ 0.9779,  0.0515,  1.3596],
        [ 1.6693, -0.4039,  2.5274],
        [ 0.5664,  0.7007,  0.4451],
        [ 0.4492,  1.2823, -0.0882],
        [-0.8840, -0.8042, -0.1665]])
"""

print(torch.add(x, y)) # Same as x + y

"""
tensor([[ 0.9779,  0.0515,  1.3596],
        [ 1.6693, -0.4039,  2.5274],
        [ 0.5664,  0.7007,  0.4451],
        [ 0.4492,  1.2823, -0.0882],
        [-0.8840, -0.8042, -0.1665]])
"""

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

"""
tensor([[ 0.9779,  0.0515,  1.3596],
        [ 1.6693, -0.4039,  2.5274],
        [ 0.5664,  0.7007,  0.4451],
        [ 0.4492,  1.2823, -0.0882],
        [-0.8840, -0.8042, -0.1665]])
"""

# adds x to y
y.add_(x)
print(y)

"""
tensor([[ 0.9779,  0.0515,  1.3596],
        [ 1.6693, -0.4039,  2.5274],
        [ 0.5664,  0.7007,  0.4451],
        [ 0.4492,  1.2823, -0.0882],
        [-0.8840, -0.8042, -0.1665]])
"""

print(x[:, 1])

"""
tensor([-0.2084, -0.7550,  0.4440,  0.9725, -1.2157])
"""

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

"""
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
"""

x = torch.randn(1)
print(x)
print(x.item())

"""
tensor([0.4004])
0.40038296580314636
"""

# NumPy Bridge

a = torch.ones(5)
print(a)

"""
tensor([1., 1., 1., 1., 1.])
"""

b = a.numpy()
print(b)

"""
[1. 1. 1. 1. 1.]
"""

a.add_(1)
print(a)
print(b)

"""
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
"""

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

"""
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
"""

# CUDA Tensors

"""We will use ``torch.device`` objects to move tensors in and out of GPU"""
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

