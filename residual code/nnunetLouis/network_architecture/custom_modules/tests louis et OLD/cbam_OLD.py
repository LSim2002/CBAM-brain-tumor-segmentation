# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:15:50 2024

@author: loulo
"""

import torch
import torch.nn as nn
from torchvision.ops import MLP
    
##A FAIRE: TORCHVISION pou le MLP de la channel attention
##checker out.view(x.size(0), -1, 1, 1))   (dimensions etc)

##INPUT OF CBAM IS For 2D data: [N, C, H, W]
## and For 3D data: [N, C, D, H, W]

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
        max_pooled = torch.max(x, **self.channel_max_op_kwargs)

        ##both avg_out and max_out will also have dimensions [N, C].
        avg_out = self.mlp(avg_pooled)
        max_out = self.mlp(max_pooled)
        ##it's standard for fully connected layers (like those in an MLP) to operate on 2D tensors where one dimension is the batch size (N) 
        ##and the other is the feature size (C in this case).

        out = avg_out + max_out
        ##dim [N, C]

        ## channel_resize_op reshapes out to have dimensions [N, C, 1, 1] for 2D inputs or [N, C, 1, 1, 1] for 3D inputs. 
        nonlin_out = self.sigmoid(self.channel_resize_op(out))

        ##on transforme le [N, C, 1, 1] (or [N, C, 1, 1, 1] for 3D), en [N, C, H, W] (or [N, C, D, H, W] for 3D) (on réplique C fois les trucs de dimension 1)
        # pour la element wise multiplication
        return nonlin_out.expand_as(x)    
    


class SpatialAttention(nn.Module):
    def __init__(self, props=None):
        super(SpatialAttention, self).__init__()
        
        ##on déclare ici la convolution (de filtre de taille 7x7x2) ainsi que la sigmoide
        self.spatial_max_op_kwargs = props['spatial_max_op_kwargs']
        self.spatial_avg_op_kwargs = props['spatial_avg_op_kwargs']

        ##on déclare la convolution
        self.conv = props['conv_op'](2, 1, kernel_size = 7, padding=3, bias=False)
        ##et la sigmoide
        self.nonlin = nn.Sigmoid()

    def forward(self, x):
        ##on utilise torch.mean et max pour obtenir le resultat des pool le long de l'axe des channels avec Keepdim=False
        avg_out = torch.mean(x,**self.spatial_mean_op_kwargs)
        max_out = torch.max(x,**self.spatial_mean_op_kwargs)
        ##on se retrouve donc avec max_out et avg_out de dimensions [N,H,W] pour des inputs 2D et de dimensions [N,D,H,W] pour des inputs 3D


        ##Unsqueeze rajoute une dimension : passe de [N,H,W] à [N,1,H,W] pour concaténer le long de la dim 1 et obtenir [N,2,H,W] (pour le cas 2D par ex)
        x_cat = torch.cat([avg_out.unsqueeze(1), max_out.unsqueeze(1)], dim=1)
        ##résultat de la concaténation  #For 2D inputs [N, 2, H, W]. For 3D inputs, it should be [N, 2, D, H, W]        

        ##on applique la convolution déclarée précédemment
        x_conv = self.conv(x_cat)
        ##The convolution takes the concatenated tensor [N, 2, H, W] (or [N, 2, D, H, W] for 3D) and outputs a tensor of [N, 1, H, W] (or [N, 1, D, H, W] for 3D), 
        ##preserving the spatial dimensions while reducing the channel dimension to 1.

        ##on applique la sigmoide et on renvoie
        return self.nonlin(x_conv)
    
    
    

class CBAM(nn.Module): ##in places est le nombre de channels par exemple du batch (un channel représente un volume ou une aire)
    def __init__(self, in_planes, ratio=16, props=None):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio, props)
        self.spatial_attention = SpatialAttention(props)

    def forward(self, x):
        x = self.channel_attention(x) * x  ##calcul de F'
        x = self.spatial_attention(x) * x  ##calcul de F''
        return x ##renvoi de F''
