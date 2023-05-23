import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import CBatchNorm1d, CResnetBlockConv1d, ResnetBlockFC


class E3Decoder(nn.Module):
    """Equivariant Decoder network."""

    def __init__(
        self,
        c_dim=128,
        vector_c_dim=16,
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
            self.add_module(f"block_{i}", ResnetBlockFC(hidden_size))

        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, p, code, **kwargs):
        n_obj, n_sample_points, _ = p.shape
        scalar_c, vector_c = code
        vector_c = vector_c.view(n_obj, self.vector_c_dim, 3).contiguous()

        p2 = torch.sum(p * p, dim=2, keepdim=True)  # (n_obj, n_sample, 1)
        x_code = torch.einsum(
            "omd,ond->omn", p, vector_c
        )  # (n_obj, n_sample, vector_c_dim)
        inv_code = torch.sum(
            scalar_c * self.scalar_mix_in(scalar_c), dim=-1, keepdim=True
        )  # (n_obj, 1)
        inv_code = inv_code.repeat(1, n_sample_points).unsqueeze(
            -1
        )  # (n_obj, n_sample, 1)
        x = torch.cat(
            [p2, x_code, inv_code], dim=-1
        )  # (n_obj, n_sample, vec_c_dim + 1 + 1)

        # sample point embedding
        x = self.point_emb(x)  # (n_obj, n_sample, hidden_size)

        # resnet blocks
        for i in range(0, self.n_blocks):
            x = self._modules[f"block_{i}"](x)

        # logit readout
        occ = self.readout(F.silu(x)).squeeze(-1)  # (o_obj, n_sample)
        return occ


class DecoderCBatchNorm(nn.Module):
    """Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    """

    def __init__(self, c_dim=128, n_blocks=4, hidden_size=64):
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
