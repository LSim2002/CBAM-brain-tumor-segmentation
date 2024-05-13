# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:15:58 2024

@author: loulo
"""
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, props=None):
        super(SpatialAttention, self).__init__()
        
        ##on déclare ici les poolings  avec keepdim=true
        self.spatial_max_op_kwargs = props['spatial_max_op_kwargs']
        self.spatial_mean_op_kwargs = props['spatial_mean_op_kwargs']

        ##on déclare la convolution (de filtre de taille 7x7x2)
        self.conv = props['conv_op'](2, 1, kernel_size = 7, padding=3, bias=False)
        ##et la sigmoide
        self.nonlin = nn.Sigmoid()


    def forward(self, x):
        ##on utilise torch.mean et max pour obtenir le resultat des pool le long de l'axe des channels avec Keepdim=True
        avg_out = torch.mean(x,**self.spatial_mean_op_kwargs)
        max_out = torch.amax(x,**self.spatial_max_op_kwargs)
        ##on se retrouve donc avec max_out et avg_out de dimensions [N,1,H,W] pour des inputs 2D et de dimensions [N,1,D,H,W] pour des inputs 3D
        ##donc plus besoin de unsqueeze
        
        x_cat = torch.cat((avg_out, max_out), dim=1)        
        ##on a mtn un truc de dim [N,2,H,W] pour des inputs 2D et de dimensions [N,2,D,H,W] pour des inputs 3D
        print('shape de la concat: ',x_cat.shape)
        
        ##on applique la convolution déclarée précédemment
        x_conv = self.conv(x_cat)
        ##The convolution  outputs a tensor of [N, 1, H, W] (or [N, 1, D, H, W] for 3D), 
        ##preserving the spatial dimensions while reducing the channel dimension to 1.

        ##on applique la sigmoide et on renvoie
        nonlin_out = self.nonlin(x_conv)
        print('shape apres conv: ',nonlin_out.shape)

        ##on transforme le [N, 1, H, W] (or [N, 1, D, H, W] for 3D), en [N, C, H, W] (or [N, C, D, H, W] for 3D) (on réplique C fois les trucs de dimension 1)
        # pour la element wise multiplication
        print('shape de loutput: ',nonlin_out.expand_as(x).shape  )  
        #print('output: ',nonlin_out.expand_as(x)  )  

        return nonlin_out.expand_as(x)    
    
    
    
dim =3

if dim == 3:
    # Test the SpatialAttention module
    N, C, D, H, W = 3, 8, 20, 20, 20  # Sample dimensions
    input_tensor = torch.randn(N, C, D, H, W)  # Sample 3D input tensor
    props = {}
    # Define properties
    props['conv_op'] = nn.Conv3d
    props['spatial_mean_op_kwargs'] =  {'dim': 1, 'keepdim': True}
    props['spatial_max_op_kwargs'] =  {'dim': 1, 'keepdim': True}

if dim == 2:
    # Test the SpatialAttention module
    N, C, H, W = 3, 8, 20, 20  # Sample dimensions
    input_tensor = torch.randn(N, C, H, W)  # Sample 3D input tensor
    props = {}
    # Define properties
    props['conv_op'] = nn.Conv2d
    props['spatial_mean_op_kwargs'] =  {'dim': 1, 'keepdim': True}
    props['spatial_max_op_kwargs'] =  {'dim': 1, 'keepdim': True}

# Create the ChannelAttention module
spatial_attention = SpatialAttention(props=props)

# Apply the module to the input tensor
output = spatial_attention(input_tensor)
# Return the dimensions of the output
print(output.shape)