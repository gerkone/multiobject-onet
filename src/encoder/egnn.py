"""Adapted from https://github.com/vgsatorras/egnn"""

import torch
from torch import nn
from torch_cluster import knn_graph

from .utils import scatter


class InstanceNorm(nn.InstanceNorm1d):
    def __init__(self, *args, is_on=False, **kwargs):
        self.is_on = is_on
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.is_on:
            x = x.transpose(0, 1)
            x = super().forward(x)
            x = x.transpose(0, 1)
        return x


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_size,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        instance_norm=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_size),
            InstanceNorm(hidden_size, affine=True, is_on=instance_norm),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size + input_nf, hidden_size),
            InstanceNorm(hidden_size, affine=True, is_on=instance_norm),
            act_fn,
            nn.Linear(hidden_size, output_nf),
        )

        layer = nn.Linear(hidden_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_size, hidden_size))
        coord_mlp.append(InstanceNorm(hidden_size, affine=True, is_on=instance_norm))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out  # (n_edges, hidden_size)

    def node_model(self, x, edge_index, edge_attr, node_attr):
        src, _ = edge_index
        agg = torch.zeros((x.shape[0], edge_attr.shape[1]), device=x.device)
        agg = agg.index_add(0, src, edge_attr)  # (n_nodes, hidden_size)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out  # (n_nodes, hidden_size)

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        src, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)  # (n_edges, 3)
        agg = torch.zeros_like(coord, device=coord.device)
        agg = agg.index_add(0, src, trans)  # (n_nodes, 3)
        # TODO (GAL) bug somewhere: edge index is not always correct
        coord = coord + agg
        return coord  # (n_nodes, 3)

    def coord2radial(self, edge_index, coord):
        src, dst = edge_index
        coord_diff = coord[src] - coord[dst]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            coord_diff = coord_diff / torch.sqrt(radial).detach() + self.epsilon

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        src, dst = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(
            h[src], h[dst], radial, edge_attr
        )  # (n_edges, hidden_size)
        coord = self.coord_model(
            coord, edge_index, coord_diff, edge_feat
        )  # (n_nodes, 3)
        h = self.node_model(
            h, edge_index, edge_feat, node_attr
        )  # (n_nodes, hidden_size)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(
        self,
        c_dim=128,
        vector_c_dim=16,
        hidden_size=64,
        scalar_dim=1,
        edge_dim=1,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=3,
        n_neighbors=5,
        residual=True,
        attention=False,
        normalize=True,
        instance_norm=True,
        eps=1e-8,
    ):
        """
        Args:
            scalar_c_dim (int): Number of features for 'h' at the output
            vector_c_dim (int): Number of features for 'x' at the output
            hidden_size (int): Number of hidden features
            scalar_dim (int): Number of features for 'h' at the input
            edge_dim (int): Number of features for the edge features
            device (str): Device (e.g. 'cpu', 'cuda:0',...)
            act_fn (str): Non-linearity
            n_layers (int): Number of layers for the EGNN
            n_neighbors (int): Number of neighbors to consider in the knn graph
            residual (bool): Use residual connections, we recommend not changing this one
            attention (bool): Whether using attention or not
            normalize (bool): Normalizes the coordinate messages.
            instance_norm (bool): Whether to use instance norm or not
            eps (float): Small number to avoid numerical instabilities
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.eps = eps

        self.embedding_in = nn.Linear(scalar_dim, self.hidden_size)

        self.c_dim = c_dim
        self.vector_c_dim = vector_c_dim

        # readout
        self.scalar_readout = nn.Linear(self.hidden_size, c_dim)
        self.readout_mix_net = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            InstanceNorm(self.hidden_size, is_on=instance_norm, affine=True),
            act_fn,
            nn.Linear(self.hidden_size, c_dim + vector_c_dim),
        )

        for i in range(0, n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL(
                    self.hidden_size,
                    self.hidden_size,
                    self.hidden_size,
                    edges_in_d=edge_dim,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    instance_norm=instance_norm,
                ),
            )
        self.to(self.device)

    def forward(self, pc: torch.Tensor, node_tag: torch.Tensor):
        """Encoder forward pass

        Args:
            inputs (torch.Tensor): point cloud (bs, n_nodes, 3)
            node_tag (torch.Tensor): node-wise instance tag (bs, n_nodes)

        Returns:
            Tuple with the scalar and the vector codes per object
        """
        bs = pc.shape[0]
        pc = pc.view(-1, 3).squeeze()  # (bs * n_nodes, 3)
        node_tag = node_tag.view(-1)  # (bs * n_nodes,)
        # feature transform
        h, x, edges, edge_attr = self._transform(pc, node_tag)
        # embedding
        h = self.embedding_in(h)  # (n_nodes, hidden_size)
        # message passing
        for i in range(0, self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)

        s_codes, v_codes = self._readout(h, x, node_tag, bs)
        return s_codes, v_codes

    def _transform(self, pc, node_tag):
        # edge indices (knn)
        # batch index per node (avoid knn across graphs)
        edges = knn_graph(pc, self.n_neighbors, node_tag)
        # scalar node features (norms)
        h = torch.norm(pc, dim=-1, keepdim=True)
        # vector node features (coordinates)
        x = pc
        # edge features (distance)
        edge_attr = torch.norm(x[edges[0]] - x[edges[1]], dim=-1, keepdim=True)
        return h, x, edges, edge_attr

    def _readout(self, h, x, node_tag, batch_size):
        # graph pooling
        n_segments = torch.unique(node_tag).shape[0]
        n_obj = n_segments // batch_size
        node_tag = node_tag.long()
        s_codes = scatter(h, node_tag, n_segments, "amax")  # (bs * n_obj, hidden_size)
        v_codes = scatter(x, node_tag, n_segments, "mean")  # (bs * n_obj, 3)

        mix = self.readout_mix_net(
            torch.cat([s_codes, torch.norm(v_codes, dim=-1, keepdim=True)], dim=-1)
        )  # (c_dim + vector_c_dim)
        scalar_mix, vector_mix = torch.split(
            mix, [self.c_dim, self.vector_c_dim], dim=-1
        )
        s_codes = self.scalar_readout(s_codes) * scalar_mix
        # vector readout
        v_codes = v_codes[:, None] * vector_mix[..., None]

        s_codes = s_codes.view(batch_size, n_obj, self.c_dim)
        v_codes = v_codes.view(batch_size, n_obj, self.vector_c_dim, 3)

        return s_codes, v_codes
