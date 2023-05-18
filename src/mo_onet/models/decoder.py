import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import ResnetBlockFC


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
            # TODO add conditional batch norm
            self.add_module(f"block_{i}", ResnetBlockFC(hidden_size))

        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, p, code, **kwargs):
        bs, n_sample_points, _ = p.shape
        scalar_c, vector_c = code
        assert scalar_c.shape[0] == vector_c.shape[0] == bs
        scalar_c = scalar_c.view(bs, self.c_dim).contiguous()
        vector_c = vector_c.view(bs, self.vector_c_dim, 3).contiguous()

        # TODO not sure what's going on here
        p2 = torch.sum(p * p, dim=2, keepdim=True)  # (bs, n_sample_points, 1)
        x_code = torch.einsum(
            "bmd,bnd->bmn", p, vector_c
        )  # (bs, n_sample_points, vector_c_dim)
        inv_code = torch.sum(
            scalar_c * self.scalar_mix_in(scalar_c), dim=-1, keepdim=True
        )
        inv_code = inv_code.repeat(1, n_sample_points).unsqueeze(
            -1
        )  # (bs, n_sample_points, 1)
        x = torch.cat(
            [p2, x_code, inv_code], dim=-1
        )  # (bs, n_sample_points, vector_c_dim + 1 + 1)

        # sample point embedding
        x = self.point_emb(x)  # (bs, n_sample_points, hidden_size)

        # resnet blocks
        for i in range(0, self.n_blocks):
            x = self._modules[f"block_{i}"](x)

        # probability readout
        occ = self.readout(F.silu(x)).squeeze(-1)  # (bs, n_sample_points)
        return occ
