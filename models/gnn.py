import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GATConv, global_mean_pool


class GNN(nn.Module):
    def __init__(
        self,
        in_channels,
        edge_attr_dim,
        hidden_channels,
        out_channels,
        dropout,
        heads,
        pretrained_weights_dir=None,
    ):
        """
        Args:
            in_channels (int): Number of input features.
            edge_attr_dim (int): Number of edge attributes.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Desired final output dimension (after attention).
            dropout (float): Dropout rate.
            heads (int): Number of attention heads.
            pretrained_weights_dir (str, optional): Path to pretrained weights. Defaults to None.
        """
        super(GNN, self).__init__()

        self.nn1 = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels * hidden_channels)
        )
        self.nn2 = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * hidden_channels)
        )

        self.conv1 = NNConv(in_channels, hidden_channels, self.nn1, aggr='mean')
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.nn2, aggr='mean')

        self.attn = GATConv(hidden_channels, out_channels, heads=heads, concat=False)

        self.dropout = nn.Dropout(dropout)

        if pretrained_weights_dir:
            self.load_state_dict(torch.load(pretrained_weights_dir))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.attn(x, edge_index)
        x = self.dropout(x)

        return global_mean_pool(x, batch)
