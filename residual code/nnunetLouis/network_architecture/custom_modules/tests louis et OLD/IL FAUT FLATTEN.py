import torch
from torchvision.ops import MLP
import torch.nn as nn

#64=16x4
input_tensor = torch.randn(1,64)
mlp = MLP(in_channels=64, 
                hidden_channels=[4,64],  # Only one hidden layer
                norm_layer=None, 
                activation_layer=nn.ReLU,
                inplace=True,
                bias=True,
                dropout=0.0)
    
output_tensor = mlp(input_tensor)

print(output_tensor.shape)

