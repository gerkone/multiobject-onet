import torch
import torch.nn as nn

# TODO decoder


class EquivariantMLP(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        padding=0.1,
    ):
        """Initialize the MLP.

        Args:
            dim (int): input dimension
            c_dim (int): dimension of latent conditioned code c
            hidden_size (int): hidden size of Decoder network
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        """
        super().__init__()

    def forward(self, p, c, **kwargs):
        pass
