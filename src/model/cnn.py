import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_layers,
                 hidden_channels,
                 kernel_size=3):
        super().__init__()
        hidden_list = []
        for i in range(hidden_layers):
            conv_layer = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', padding_mode='circular')
            hidden_list.append(conv_layer)
            hidden_list.append(nn.GELU())
            
        self.model = nn.Sequential(
            nn.Conv2d(input_size, hidden_channels, kernel_size, padding='same', padding_mode='circular'),
            nn.ReLU(),
            *hidden_list,
            nn.Conv2d(hidden_channels, input_size, kernel_size, padding='same', padding_mode='circular')
        )
        # for layer in self.model:
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.normal_(layer.weight, std = 0.001)
        #         nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)