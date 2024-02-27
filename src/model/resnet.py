import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self,
                 channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding='same', padding_mode='circular')
        self.conv2 = nn.Conv2d(channels, channels, 3, padding='same', padding_mode='circular')
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x + residual
        x = self.activation(x)
        return x

class SimpleResnet(nn.Module):
    def __init__(self,
                 input_dim, 
                 resblock_layers,
                 resblock_channels):
        super().__init__()

        resblock_list = [ResBlock(resblock_channels) for i in range(resblock_layers)]
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, resblock_channels, 3, padding='same', padding_mode='circular'),
            nn.GELU(),
            *resblock_list,
            nn.Conv2d(resblock_channels, input_dim, 3, padding='same', padding_mode='circular')
        )
        
        
        # initialize layers
        nn.init.normal_(self.input_layer.weight, std = 0.001)
        nn.init.zeros_(self.input_layer.bias)
        
        nn.init.normal_(self.hidden_layer.weight, std = 0.001)
        nn.init.zeros_(self.hidden_layer.bias)

        nn.init.normal_(self.output_layer.weight, std = 0.001)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        return self.model(x)