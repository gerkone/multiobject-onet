import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

from src.layers import CBatchNorm1d, CResnetBlockConv1d, ResnetBlockFC


class E3Decoder(nn.Module):
    """Equivariant Decoder network."""

    def __init__(
        self,
        c_dim=32,
        n_blocks=2,
        hidden_size=64,
    ):
        """Initialize the MLP.

        Args:
            dim (int): input dimension
            c_dim (int): dimension of latent conditioned code c
            hidden_size (int): hidden size of Decoder network
            padding (float): conventional padding paramter of ONet for unit cube
        """
        super().__init__()

        self.c_dim = c_dim

        self.scalar_mix_in = nn.Linear(c_dim, c_dim)
        self.point_emb = nn.Linear(1 + 1 + 1, hidden_size)

        self.n_blocks = n_blocks

        for i in range(0, n_blocks):
            self.add_module(f"block_{i}", CResnetBlockConv1d(c_dim, hidden_size))

        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, p, code, **kwargs):
        bs, n_sample_points, _ = p.shape
        vector_c, scalar_c = code
        bs, n_obj, _ = scalar_c.shape

        p2 = torch.sum(p * p, dim=2, keepdim=True).repeat(
            n_obj, 1, 1
        )  # (bs * o_obj, n_sample, 1)
        x_code = torch.einsum("bmd,bod->bom", p, vector_c)  # (bs, o_obj, n_sample)
        x_code = x_code.reshape(bs * n_obj, n_sample_points, -1).contiguous()
        scalar_c = scalar_c.view(bs * n_obj, self.c_dim).contiguous()
        vector_c = vector_c.view(bs * n_obj, 1, 3).contiguous()

        inv_code = torch.sum(
            scalar_c * self.scalar_mix_in(scalar_c), dim=-1, keepdim=True
        )  # (bs * o_obj, 1)
        inv_code = inv_code.repeat(1, n_sample_points)
        inv_code = inv_code[..., None]  # (bs * o_obj, n_sample, 1)
        x = torch.cat(
            [p2, x_code, inv_code], dim=-1
        )  # (bs * o_obj, n_sample, 1 + 1 + 1)

        # sample point embedding
        x = self.point_emb(x)
        x = x.transpose(1, 2).contiguous()  # (bs * o_obj, hidden_size, n_sample)

        # resnet blocks
        for i in range(0, self.n_blocks):
            x = self._modules[f"block_{i}"](x, scalar_c)

        x = x.transpose(1, 2).contiguous()  # (bs * o_obj, n_sample, hidden_size)

        # logit readout
        occ = self.readout(F.silu(x)).squeeze(-1)  # (bs * o_obj, n_sample)
        occ = occ.view(bs, n_obj, n_sample_points)  # (bs, o_obj, n_sample)
        return occ


class CBNDecoder(nn.Module):
    """Decoder with conditional batch normalization.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    """

    def __init__(self, c_dim=32, n_blocks=4, hidden_size=64):
        super().__init__()

        self.n_blocks = n_blocks
        self.c_dim = c_dim

        self.fc_p = nn.Conv1d(3, hidden_size, 1)

        for i in range(0, n_blocks):
            self.add_module(f"block_{i}", CResnetBlockConv1d(c_dim, hidden_size))

        self.bn = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        self.actvn = F.relu

    def forward(self, p, code, **kwargs):
        if isinstance(code, tuple):
            code = code[1]

        bs, n_obj, _ = code.shape

        # add n_obj dimension to p
        p = p.repeat(n_obj, 1, 1)  # (bs * n_obj, n_points, 3)
        p = p.transpose(1, 2)
        code = code.flatten(0, 1)  # (bs * n_obj, c_dim)

        h_p = self.fc_p(p)

        for i in range(0, self.n_blocks):
            h_p = self._modules[f"block_{i}"](h_p, code)

        out = self.fc_out(self.actvn(self.bn(h_p, code)))
        out = out.view(bs, n_obj, -1)  # (bs, n_obj, n_points)

        return out


class DGCNNDecoder(nn.Module):
    """Dynamic graph convolution decoder.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        n_neighbors (int): number of neighbors to consider in knn graph
    """

    def __init__(
        self, c_dim=24, hidden_size=128, leaky=False, n_blocks=5, n_neighbors=20
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.k = n_neighbors

        self.fc_c = nn.ModuleList(
            [nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)]
        )

        self.fc_p = nn.Linear(3, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for _ in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6 + c_dim, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size, c_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, p, codes, **kwargs):
        bs, n_points, _ = p.shape

        node_tag = kwargs.get("node_tag", torch.zeros(bs * n_points, dtype=torch.long))
        node_tag = node_tag.flatten()

        n_obj = (node_tag.max() + 1) // bs

        pc, feat = codes
        pc = pc.permute(0, 2, 1)  # (bs, 3, n_pc)
        feat = feat.permute(0, 2, 1)  # (bs, c_dim, n_pc)

        # if n_points >= 30000:
        #     c_list = []
        #     for p_split in torch.split(p, 10000, dim=1):
        #         raise NotImplementedError
        # else:
        p = p.permute(0, 2, 1)
        edge, x, _, yfeat, idx = self._graph_feats(p, pc, feat)

        x = torch.cat([edge, x, yfeat], dim=1)  # (bs, 3 + 3 + hidden, N, k)
        x = self.conv1(x)  # (bs, hidden_size, N, k)
        x = self.conv2(x)
        x = self.conv3(x)
        # neighbor pool
        c = x.max(dim=-1, keepdim=False)[0]  # (bs, 24, N)
        c = c.permute(0, 2, 1)
        p = p.permute(0, 2, 1)

        p = p.float()
        net = self.fc_p(p)  # (bs, n_points, hidden_size)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        occ = self.fc_out(self.actvn(net))
        occ = occ.squeeze(-1)  # (bs, n_points)

        if self.training:
            # object-wise occupancy assignment from nearest object to grid point
            grid_to_obj = node_tag[idx.view(bs, n_points, self.k)[..., 0]]
            obj_occ = torch.zeros((bs * n_obj, n_points), device=p.device)
            obj_occ.scatter_add_(0, grid_to_obj, occ)
            obj_occ = obj_occ.view(bs, n_obj, n_points)
            assert (obj_occ.sum(1) - occ).sum() < 1e-8
            occ = obj_occ

        return occ  # (bs, n_obj, n_points)

    def _graph_feats(self, x, y, yfeat):
        bs, _, n_points_x = x.shape
        n_points_x = x.shape[2]
        n_points_y = y.shape[2]

        k = self.k

        x = x.permute(0, 2, 1).view(bs * n_points_x, -1)
        y = y.permute(0, 2, 1).view(bs * n_points_y, -1)
        yfeat = yfeat.permute(0, 2, 1).view(bs * n_points_y, -1)

        batch_x = torch.arange(bs).repeat_interleave(n_points_x).to(x.device)
        batch_y = torch.arange(bs).repeat_interleave(n_points_y).to(x.device)

        # idx of shape (n_points_x,) with the index of the nearest center
        src, dst = knn(y, x, k, batch_x=batch_y, batch_y=batch_x)

        y = y[dst, :]
        yfeat = yfeat[dst, :]
        edge = y - x[src, :]

        y = y.view(bs, n_points_x, k, 3).permute(0, 3, 1, 2)  # (bs, 3, N, k)
        yfeat = yfeat.view(bs, n_points_x, k, -1).permute(0, 3, 1, 2)  # (bs, d, N, k)
        edge = edge.view(bs, n_points_x, k, 3).permute(0, 3, 1, 2)  # (bs, 3, N, k)
        x = x[src, :].view(bs, n_points_x, k, 3).permute(0, 3, 1, 2)  # (bs, 3, N, k)

        return edge, x, y, yfeat, dst
