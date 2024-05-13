# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:15:58 2024

@author: loulo
"""
import torch
import torch.nn as nn
from torchvision.ops import MLP



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, props=None):
        super(ChannelAttention, self).__init__()

        # Store pooling dimensions ##on ne peut pas definir le torch.mean ici, x doit lui etre donné
        self.channel_mean_op_kwargs = props['channel_mean_op_kwargs']
        self.channel_max_op_kwargs = props['channel_max_op_kwargs']

        ##on importe le resizing duepuis les props: resize en [N, C, 1, 1] pour inputs 2d et en [N, C, D, H, W] pour inputs 3d
        self.channel_resize_op = props['channel_resize_op']
        ##reszing qui sera fait en sortie du module de channel attention

        # MLP definition ##planes is another word for channels
        hidden_planes = in_planes // ratio
        # Using torchvision's MLP
        ##activation function is applied after each linear layer (except for the final linear layer). 
        ## each linear layer within the MLP includes a bias term.  
        self.mlp = MLP(in_channels=in_planes, 
                    hidden_channels=[hidden_planes,in_planes],  # Only one hidden layer followed by output layer of size in_planes
                    norm_layer=None, 
                    activation_layer=nn.ReLU,
                    inplace=True,
                    bias=True,
                    dropout=0.0)
        
        ##la sigmoide pour l'operation finale
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ##dim de torch.mean avec keepdim=True: For 2d : [N, C, 1, 1]. For 3D inputs, they will be [N, C, 1, 1, 1].
        ##ICI on a keepdim = False ! Donc For 2d and 3d : [N, C].
        avg_pooled = torch.mean(x, **self.channel_mean_op_kwargs)
        max_pooled = torch.amax(x, **self.channel_max_op_kwargs)
        
        print('post pool: ', avg_pooled.shape)

        ##both avg_out and max_out will also have dimensions [N, C].
        avg_out = self.mlp(avg_pooled)
        max_out = self.mlp(max_pooled)
        ##it's standard for fully connected layers (like those in an MLP) to operate on 2D tensors where one dimension is the batch size (N) 
        ##and the other is the feature size (C in this case).
        print('post mlp shape: ', avg_out.shape)

        out = avg_out + max_out
        ##dim [N, C]

        ## channel_resize_op reshapes out to have dimensions [N, C, 1, 1] for 2D inputs or [N, C, 1, 1, 1] for 3D inputs. 
        nonlin_out = self.sigmoid(self.channel_resize_op(out))

        ##on transforme le [N, C, 1, 1] (or [N, C, 1, 1, 1] for 3D), en [N, C, H, W] (or [N, C, D, H, W] for 3D) (on réplique C fois les trucs de dimension 1)
        # pour la element wise multiplication
        print('shape de loutput: ', nonlin_out.expand_as(x).shape)
        #print('output: ',nonlin_out.expand_as(x)  )  

        return nonlin_out.expand_as(x)    
    
    
dim =3

if dim == 3:
    # Test the ChannelAttention module
    N, C, D, H, W = 3, 32, 20, 20, 20  # Sample dimensions
    input_tensor = torch.randn(N, C, D, H, W)  # Sample 3D input tensor
    
    # Define properties
    props = {
        'channel_mean_op_kwargs': {'dim': (2, 3, 4), 'keepdim': False},  # For 3D: Mean across D, H, W dimensions
        'channel_max_op_kwargs': {'dim': (2, 3, 4), 'keepdim': False},  # For 3D: Max across D, H, W dimensions
        'channel_resize_op': lambda x: x.view(x.size(0), x.size(1), 1, 1, 1)  # Resize for 3D inputs
    }

if dim == 2:
    # Test the ChannelAttention module
    N, C, H, W = 3, 32, 20, 20  # Sample dimensions
    input_tensor = torch.randn(N, C, H, W)  # Sample 3D input tensor
    
    # Define properties
    props = {
        'channel_mean_op_kwargs': {'dim': (2,3), 'keepdim': False},  # For 3D: Mean across D, H, W dimensions
        'channel_max_op_kwargs': {'dim': (2,3), 'keepdim': False},  # For 3D: Max across D, H, W dimensions
        'channel_resize_op': lambda x: x.view(x.size(0), x.size(1), 1, 1)  # Resize for 3D inputs
    }

# Create the ChannelAttention module
channel_attention = ChannelAttention(in_planes=C, props=props)

# Apply the module to the input tensor
output = channel_attention(input_tensor)
# Return the dimensions of the output
