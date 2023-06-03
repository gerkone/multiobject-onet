import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

from src.layers import CBatchNorm1d, CResnetBlockConv1d, ResnetBlockFC
from src.encoder.utils import scatter
from src.encoder.egnn import E_GCL


class E3Decoder(nn.Module):
    """Equivariant Decoder network."""

    def __init__(
        self,
        c_dim=32,
        vector_c_dim=4,
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
        self.vector_c_dim = vector_c_dim

        self.scalar_mix_in = nn.Linear(c_dim, c_dim)
        self.point_emb = nn.Linear(vector_c_dim + 1 + 1, hidden_size)

        self.n_blocks = n_blocks

        for i in range(0, n_blocks):
            self.add_module(f"block_{i}", CResnetBlockConv1d(c_dim, hidden_size))

        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, p, code, **kwargs):
        bs, n_sample_points, _ = p.shape
        scalar_c, vector_c = code
        bs, n_obj, _ = scalar_c.shape

        p2 = torch.sum(p * p, dim=2, keepdim=True).repeat(
            n_obj, 1, 1
        )  # (bs * o_obj, n_sample, 1)
        x_code = torch.einsum(
            "bmd,bond->bomn", p, vector_c
        )  # (bs, o_obj, n_sample, vector_c_dim)
        x_code = x_code.reshape(
            bs * n_obj, n_sample_points, self.vector_c_dim
        ).contiguous()
        scalar_c = scalar_c.view(bs * n_obj, self.c_dim).contiguous()
        vector_c = vector_c.view(bs * n_obj, self.vector_c_dim, 3).contiguous()

        inv_code = torch.sum(
            scalar_c * self.scalar_mix_in(scalar_c), dim=-1, keepdim=True
        )  # (bs * o_obj, 1)
        inv_code = inv_code.repeat(1, n_sample_points)
        inv_code = inv_code[..., None]  # (bs * o_obj, n_sample, 1)
        x = torch.cat(
            [p2, x_code, inv_code], dim=-1
        )  # (bs * o_obj, n_sample, vec_c_dim + 1 + 1)

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
            code = code[0]

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
    """Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    """

    def __init__(
        self, dim=3, c_dim=32, hidden_size=256, leaky=False, n_blocks=5, n_neighbors=20
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.k = n_neighbors

        self.fc_c = nn.ModuleList(
            [nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)

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

        node_tag = kwargs.get("node_tag", None)

        n_obj = (node_tag.max() + 1) // bs

        feat, pc = codes
        pc = pc.permute(0, 2, 1)  # (bs, 3, n_pc)
        feat = feat.squeeze().permute(0, 2, 1)  # (bs, c_dim, n_pc)

        if n_points >= 30000:
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                raise NotImplementedError
        else:
            p = p.permute(0, 2, 1)
            edge, x, _, yfeat = self._graph_feats(p, pc, feat, self.k, batch=node_tag)
            x = torch.cat([edge, x, yfeat], dim=1)  # (bs, 20 * 2 * 3, n_pc, k)
            x = self.conv1(x)  # (bs, hidden_size, n_pc, k)
            x = self.conv2(x)
            x = self.conv3(x)
            # neighboor pool
            c = x.max(dim=-1, keepdim=False)[0]  # (bs, 64, n_pc)
            c = c.permute(0, 2, 1)
            p = p.permute(0, 2, 1)

        obj_codes = scatter(c.flatten(0, 1), node_tag, bs * n_obj, "amax")
        obj_codes = obj_codes.view(bs, n_obj, -1)  # (bs, n_obj, c_dim)

        p = p.float()
        net = self.fc_p(p)  # (bs, n_points, hidden_size)
        # object wise points
        net = net.unsqueeze(1).repeat(
            1, n_obj, 1, 1
        )  # (bs, n_obj, n_points, hidden_size)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](obj_codes).unsqueeze(-2)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out  # (bs, n_obj, n_points)

    def _graph_feats(self, x, y, yfeat, k=20, batch=None):
        """
        used when target has extra feature dims
        :param x: (B, 3, N1)
        :param y: (B, 3, N2)
        :param yfeat: (B, d, N2)
        :param k:
        :return:
        """
        bs, _, n_points_x = x.shape
        n_points_y = y.shape[2]
        n_obj = (batch.max() + 1) // bs
        batch = batch.flatten()

        x = x.permute(0, 2, 1).view(bs * n_points_x, -1)
        y = y.permute(0, 2, 1).view(bs * n_points_y, -1)
        yfeat = yfeat.permute(0, 2, 1).view(bs * n_points_y, -1)

        x_objwise = x.repeat(n_obj, 1)

        batch_points = (
            torch.arange(bs * n_obj)
            .view(bs * n_obj, 1)
            .repeat(1, n_points_x)
            .view(-1)
            .to(x.device)
        )

        # TODO GAL neigbourhood size should not depend on batch size and number of objects
        k = min(k, batch.bincount().min().item() - 1)
        snd, dst = knn(
            x_objwise, y, k, batch_x=batch_points, batch_y=batch
        )  # (num_points * k,)

        y = y[snd, :]
        yfeat = yfeat[snd, :]
        edge = y - x_objwise[dst, :]

        y = y.view(bs, n_points_y, k, 3).permute(0, 3, 1, 2)  # (bs, 3, n_pc, k)
        yfeat = yfeat.view(bs, n_points_y, k, -1).permute(
            0, 3, 1, 2
        )  # (bs, d, n_pc, k)
        edge = edge.view(bs, n_points_y, k, 3).permute(0, 3, 1, 2)  # (bs, 3, n_pc, k)
        x = (
            x_objwise[dst, :].view(bs, n_points_y, k, 3).permute(0, 3, 1, 2)
        )  # (bs, 3, num_points, k)

        return edge, x, y, yfeat
