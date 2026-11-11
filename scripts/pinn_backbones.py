"""Literature-inspired dense backbones for the conditional nozzle PINN."""

import math

import torch
from torch import nn


class LiteraturePINN(nn.Module):
    """A tanh FNN with optional fixed multiscale temporal Fourier features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        time_column: int,
        *,
        use_fourier: bool,
    ) -> None:
        super().__init__()
        self.time_column = time_column
        frequencies = None
        if use_fourier:
            scales = torch.tensor([0.5, 2.0, 8.0])[:, None]
            frequencies = (scales * torch.randn(3, 16)).reshape(1, -1)
            input_dim += 2 * frequencies.numel()
        self.register_buffer("frequencies", frequencies)

        widths = [input_dim, *([hidden_dim] * num_blocks), output_dim]
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths, widths[1:])):
            layer = nn.Linear(left, right)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if index < num_blocks:
                layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.frequencies is not None:
            time = inputs[:, self.time_column : self.time_column + 1]
            phase = 2.0 * math.pi * time * self.frequencies
            inputs = torch.cat((inputs, torch.sin(phase), torch.cos(phase)), dim=1)
        return self.layers(inputs)
