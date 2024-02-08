import torch.nn as nn

from typing import Optional

class MLP(nn.Module):
    def __init__(self, output_size, hidden_size, input_size: Optional[int] = None, hidden_layers=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        if self.input_size is None:
            self.input_layer = nn.LazyLinear(hidden_size)
        else:
            self.input_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers-1)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

        if self.input_size is not None:
            # initialize layers
            nn.init.normal_(self.input_layer.weight, std = 0.001)
            nn.init.zeros_(self.input_layer.bias)
            
            for layer in self.hidden_layers:
                nn.init.normal_(layer.weight, std = 0.001)
                nn.init.zeros_(layer.bias)

            nn.init.normal_(self.output_layer.weight, std = 0.001)
            nn.init.zeros_(self.output_layer.bias)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

        # initialize layers
        nn.init.normal_(self.input_layer.weight, std = 0.001)
        nn.init.zeros_(self.input_layer.bias)
        
        nn.init.normal_(self.hidden_layer.weight, std = 0.001)
        nn.init.zeros_(self.hidden_layer.bias)

        nn.init.normal_(self.output_layer.weight, std = 0.001)
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x + input_tensor