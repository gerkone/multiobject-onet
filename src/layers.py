from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(nn.Module):
    """Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
    """

    def __init__(self, c_dim, size_in, size_h=None, size_out=None, norm_method="batch"):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(nn.Module):
    """Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    """

    def __init__(self, c_dim, f_dim, norm_method="batch"):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_mix = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == "batch":
            # TODO (GAL): track_running_stats=False
            self.bn = nn.BatchNorm1d(f_dim, affine=False, track_running_stats=False)
        elif norm_method == "instance":
            self.bn = nn.InstanceNorm1d(f_dim, affine=False, track_running_stats=False)
        else:
            raise ValueError("Invalid normalization method!")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)
        nn.init.uniform_(self.conv_mix.weight)
        nn.init.zeros_(self.conv_mix.bias)

    def forward(self, x, c):
        assert c.size(-1) == self.c_dim

        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)  # (bs, f_dim, 1)
        beta = self.conv_beta(c)

        # break object-wise symmetry
        # mix = self.conv_mix(c)  # (bs, f_dim, 1)

        # TODO (GAL): fix shapes with batch here

        # Batchnorm
        x = self.bn(x)
        out = gamma * x + beta  # * mix  # (bs * n_obj, f_dim, n_points)
        return out


# TODO (GAL)


class E3ResnetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

    def forward(self, x):
        return x


class E3Linear(nn.Module):
    def __init__(self, n_sin: int, n_vin: int, n_sout: int, n_vout: int):
        super().__init__()

        self.n_sout = n_sout
        self.n_vout = n_vout

        self.scalar_down = nn.Linear(n_sin, n_sout)
        self.vector_down = nn.Linear(n_vin, n_vout)
        self.mix = nn.Linear(n_sin + 1, n_sout + n_vout, bias=False)

    def forward(self, s, v):
        mix = self.mix(torch.cat([s, torch.norm(v, dim=-1)], dim=-1))
        mix_l, mix_r = torch.split(mix, [self.n_sout, self.n_vout], -1)
        ds = self.scalar_down(s) * mix_l
        dv = self.vector_down(s) * mix_r
        return s + ds, v + dv


class E3GatedBlock(nn.Module):
    """
    Gated equivariant block.
    Transforms scalar and vector representation using gated nonlinearities.

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021 (to appear)

    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation=nn.SiLU,
        sactivation=None,
    ):
        """
        Args:
            n_sin: number of input scalar features
            n_vin: number of input vector features
            n_sout: number of output scalar features
            n_vout: number of output vector features
            n_hidden: number of hidden units
            activation: interal activation function
            sactivation: activation function for scalar outputs
        """
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = nn.Linear(n_vin, 2 * n_vout, bias=False)
        self.scalar_net = nn.Sequential(
            nn.Linear(n_sin + n_vout, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_sout + n_vout),
        )
        self.sactivation = sactivation

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        scalars, vectors = inputs
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out
