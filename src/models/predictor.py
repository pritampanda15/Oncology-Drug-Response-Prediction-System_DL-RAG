"""
src/models/predictor.py

Feed-forward neural network for drug response prediction.
Optimized with regularization to prevent overfitting.
"""

import torch
import torch.nn as nn


class DrugResponsePredictor(nn.Module):
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = None,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)