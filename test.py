import torch

from ladybirdmnist.datasets import LadybirdMNIST

dataset = LadybirdMNIST(root='./data', train=True, download=True)

print(len(dataset.data[0]))

