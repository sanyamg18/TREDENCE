"""
Edhas - Self-Pruning Neural Network
Core model implementation with learnable gate parameters
Author: Kudrat Anand (RA2311026010844)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable pruning gates.
    Gates use sigmoid to produce soft masks during training.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable gate scores - same shape as weight matrix
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
    def forward(self, x):
        # Compute soft gates using sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        pruned_weights = self.weight * gates
        
        # Standard linear transformation with pruned weights
        return F.linear(x, pruned_weights, self.bias)
    
    def get_gate_values(self):
        """Get current gate values (after sigmoid)"""
        return torch.sigmoid(self.gate_scores)
    
    def get_sparsity(self, threshold=0.5):
        """Calculate sparsity percentage based on gate threshold"""
        gates = self.get_gate_values()
        pruned_count = (gates < threshold).sum().item()
        total_count = gates.numel()
        return (pruned_count / total_count) * 100


class GreenSparseNet(nn.Module):
    """
    Self-pruning neural network for sustainable AI.
    Uses PrunableLinear layers throughout.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_hidden_layers=2):
        super(GreenSparseNet, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(PrunableLinear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(PrunableLinear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = PrunableLinear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through hidden layers with ReLU activation
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Output layer (no activation - CrossEntropyLoss handles it)
        x = self.output_layer(x)
        return x
    
    def get_total_sparsity(self, threshold=0.5):
        """Get overall model sparsity percentage"""
        total_params = 0
        pruned_params = 0
        
        for layer in self.layers + [self.output_layer]:
            gates = layer.get_gate_values()
            total_params += gates.numel()
            pruned_params += (gates < threshold).sum().item()
        
        return (pruned_params / total_params) * 100 if total_params > 0 else 0
    
    def get_gate_l1_loss(self):
        """Compute L1 sparsity loss across all gates"""
        l1_loss = 0
        for layer in self.layers + [self.output_layer]:
            l1_loss += torch.sum(layer.get_gate_values())
        return l1_loss
    
    def count_parameters(self, count_pruned=True):
        """Count model parameters"""
        if count_pruned:
            # Count all parameters (including pruned)
            return sum(p.numel() for p in self.parameters())
        else:
            # Count only active (non-pruned) parameters
            active_params = 0
            for layer in self.layers + [self.output_layer]:
                gates = layer.get_gate_values()
                active = (gates >= 0.5).sum().item()
                active_params += active
            return active_params
