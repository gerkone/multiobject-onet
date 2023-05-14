"""Adapted from https://github.com/vgsatorras/egnn"""

import torch
from torch import nn
from torch_geometric.nn import knn_graph
from torch_kmeans import SoftKMeans

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_dim,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + input_nf, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_nf),
        )

        layer = nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_dim, hidden_dim))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        src, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, src, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        src, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, src, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, src, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)
        coord = coord + agg
        return coord

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
        edge_feat = self.edge_model(h[src], h[dst], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNNLocal(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_dim=64,
        padding=0.1,
        scalar_dim=1,
        edge_dim=1,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        n_neighbors=5,
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="sum",
        tanh=False,
        eps=1e-8,
    ):
        """
        :param dim: input point dimension
        :param c_dim: Number of features for 'h' at the output
        :param hidden_dim: Number of hidden features
        :param padding: conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        :param scalar_dim: Number of features for 'h' at the input
        :param edge_dim: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param n_neighbors: Number of neighbors to consider in the knn graph
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        :param eps: Small number to avoid numerical instabilities
        """
        super().__init__()
        _ = dim, padding
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.eps = eps
        # TODO should not be fixed
        self.n_objects = 5

        self.embedding_in = nn.Linear(scalar_dim, self.hidden_dim)
        self.scalar_embedding_out = nn.Linear(self.hidden_dim, c_dim)
        self.vector_embedding_out = nn.Linear(1, c_dim)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    edges_in_d=edge_dim,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    coords_agg=coords_agg,
                    tanh=tanh,
                ),
            )
        self.to(self.device)

    def forward(self, pc, ):
        bs, n_nodes, _ = pc.shape
        # feature transform
        h, x, edges, edge_attr = self._transform(pc)
        # embedding
        h = self.embedding_in(h)
        # message passing
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        # outputs
        scalar_node_embeddings = self.scalar_embedding_out(h).view(bs, n_nodes, -1)
        gate = self.vector_embedding_out(torch.norm(x, dim=-1, keepdim=True))[..., None]
        vector_node_embeddings = (x[:, None] * gate).view(bs, n_nodes, -1)
        node_embeddings = (scalar_node_embeddings, vector_node_embeddings)
        node_tag = SoftKMeans(
            init_method="k-means++", n_clusters=self.n_objects, verbose=False
        )(scalar_node_embeddings).labels
        return node_embeddings, node_tag
    
    def _transform(self, pc):
        # edge indices (knn)
        bs, n_nodes, _ = pc.shape
        # flatten batch and nodes
        pc = pc.view(bs * n_nodes, -1)
        batch = torch.arange(0, bs, device=pc.device)
        batch = batch.repeat_interleave(n_nodes).long()
        edges = knn_graph(pc, self.n_neighbors, batch)
        # scalar node features (norms)
        h = torch.norm(pc, dim=-1, keepdim=True)
        # vector node features (coordinates)
        x = pc
        # edge features (distance)
        edge_attr = torch.norm(x[edges[0]] - x[edges[1]], dim=-1, keepdim=True)
        return h, x, edges, edge_attr


# TODO replace these with torch_scatter
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
