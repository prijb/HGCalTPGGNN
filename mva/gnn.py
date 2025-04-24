import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm


class SimpleGNN(nn.Module):
    def __init__(self, in_channels=14, global_features=3):
        """
        in_channels: Number of node features.
        hidden_channels: Hidden layer dimension.
        global_features: Number of additional global features per graph.
        The model outputs a single logit per graph.
        """
        super(SimpleGNN, self).__init__()
        # Each layer is a conv and graph norm
        self.layer1 = nn.Sequential(
            GCNConv(in_channels, 32),
            GraphNorm(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            GCNConv(32, 64),
            GraphNorm(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            GCNConv(64, 128),
            GraphNorm(128),
            nn.ReLU()
        )
        self.linear = nn.Linear(128 + global_features, 1)

    def forward(self, x, edge_index, batch, u):
        # First layer
        x = self.layer1[0](x, edge_index)  # GCNConv
        x = self.layer1[1](x)              # GraphNorm
        x = self.layer1[2](x)              # ReLU
        
        # Second layer
        x = self.layer2[0](x, edge_index)  # GCNConv
        x = self.layer2[1](x)              # GraphNorm
        x = self.layer2[2](x)              # ReLU
        
        # Third layer
        x = self.layer3[0](x, edge_index)  # GCNConv
        x = self.layer3[1](x)              # GraphNorm
        x = self.layer3[2](x)              # ReLU


        x_pool = global_mean_pool(x, batch)
        x_cat = torch.cat([x_pool, u], dim=1)
        out = self.linear(x_cat)
        
        #out = torch.sigmoid(out) # Uncomment if using BCELoss instead of BCEWithLogitsLoss
        return out
    