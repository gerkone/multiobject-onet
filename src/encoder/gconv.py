import torch
import torch.nn as nn
from torch_cluster import knn_graph

from .utils import scatter


class MOGConv(nn.Module):
    def __init__(self, c_dim=128, hidden_size=128, n_neighbors=20):
        super().__init__()
        self.c_dim = c_dim
        assert n_neighbors < 100, "n_neighbors should be less than 100"
        self.n_neighbors = n_neighbors

        self.bn1 = nn.InstanceNorm2d(hidden_size)
        self.bn2 = nn.InstanceNorm2d(hidden_size)
        self.bn3 = nn.InstanceNorm2d(hidden_size)
        self.bn4 = nn.InstanceNorm2d(hidden_size)
        self.bn5 = nn.InstanceNorm2d(hidden_size)
        self.bn6 = nn.InstanceNorm2d(hidden_size)
        self.bn7 = nn.InstanceNorm2d(hidden_size)
        self.bn8 = nn.InstanceNorm2d(hidden_size)
        self.bn9 = nn.InstanceNorm1d(hidden_size * 4)
        self.bn10 = nn.InstanceNorm1d(hidden_size * 4)
        self.bn11 = nn.InstanceNorm1d(hidden_size * 2)
        self.bn12 = nn.InstanceNorm1d(hidden_size * 2)
        self.bn13 = nn.InstanceNorm1d(hidden_size * 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv9 = nn.Sequential(
            nn.Conv1d(hidden_size * 4, hidden_size * 2, kernel_size=1, bias=False),
            self.bn9,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv10 = nn.Sequential(
            nn.Conv1d(hidden_size * 3, hidden_size * 2, kernel_size=1, bias=False),
            self.bn10,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv11 = nn.Sequential(
            nn.Conv1d(hidden_size * 3, hidden_size * 2, kernel_size=1, bias=False),
            self.bn11,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(hidden_size * 3, hidden_size, kernel_size=1, bias=False),
            self.bn12,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=1, bias=False),
            self.bn13,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.readout = nn.Conv1d(hidden_size, c_dim, kernel_size=1, bias=True)

    def forward(self, pc, node_tag):
        """MOGConv forward.

        Args:
            x (torch.Tensor): Point cloud (bs, n_nodes, 3)
            node_tag (torch.Tensor): Node tag index for aggregation (bs, n_nodes)
        """
        bs, n_nodes, _ = pc.shape
        n_obj = node_tag[0].max().item() + 1
        node_tag = node_tag.view(-1)  # (bs * n_nodes,)

        x = pc.clone()
        x = x.view(-1, 3).squeeze()  # (bs * n_nodes, 3)

        k = self.n_neighbors

        x, idx = self._transform(x, k, node_tag)  # (3, bs * n_nodes, k)
        x = self.conv1(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv2(x)
        x = x.permute(1, 2, 0)
        x1 = scatter(x, idx, bs * n_nodes, "amax")  # (bs * n_nodes, hidden_size)

        x, idx = self._transform(x1, k, node_tag)  # (hidden_size, bs * n_nodes)
        x = self.conv3(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv4(x)
        x = x.permute(1, 2, 0)
        x2 = scatter(x, idx, bs * n_nodes, "amax") + x1  # (hidden_size, bs * n_nodes)

        x, idx = self._transform(x2, k, node_tag)  # (hidden_size, bs * n_nodes)
        x = self.conv5(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv6(x)
        x = x.permute(1, 2, 0)
        x3 = scatter(x, idx, bs * n_nodes, "amax") + x2  # (bs * n_nodes, hidden_size)

        x, idx = self._transform(x3, k, node_tag)  # (hidden_size, bs * n_nodes)
        x = self.conv7(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv8(x)
        x = x.permute(1, 2, 0)
        x4 = scatter(x, idx, bs * n_nodes, "amax") + x3  # (bs * n_nodes, hidden_size)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.transpose(0, 1)  # (hidden_size * 4, bs * n_nodes, 1)

        x = self.conv9(x)  # (hidden_size * 2, bs * n_nodes)
        x = x.max(dim=-1, keepdim=True)[0]  # (hidden_size * 2, 1)

        x = x.repeat(1, bs * n_nodes)  # (hidden_size * 2, bs * n_nodes)

        x4 = torch.cat((x, x4.T), dim=0)  # (hidden_size * 3, bs * n_nodes)
        x4 = self.conv10(x4)  # (hidden_size * 2, bs * n_nodes)

        x3 = torch.cat((x4, x3.T), dim=0)  # (hidden_size * 3, bs * n_nodes)
        x3 = self.conv11(x3)  # (hidden_size * 2, bs * n_nodes)

        x2 = torch.cat((x3, x2.T), dim=0)  # (hidden_size * 3, bs * n_nodes)
        x2 = self.conv12(x2)  # (hidden_size, bs * n_nodes)

        x1 = torch.cat((x2, x1.T), dim=0)  # (hidden_size * 2, bs * n_nodes)
        x1 = self.conv13(x1)  # (hidden_size, bs * n_nodes)

        # readout to global code
        code = self.readout(x1).permute(1, 0)  # (bs * n_nodes, c_dim)
        codes = code.view(bs, n_nodes, -1)  # (bs, n_nodes, c_dim)
        # collect codes on objects
        # codes = scatter(code, node_tag.long(), bs * n_obj, "mean")  # (n_obj, c_dim)
        # codes = codes.view(bs, n_obj, -1)  # (bs, n_obj, c_dim)

        return codes, pc

    def _transform(self, x, k, batch):
        with torch.no_grad():
            if self.n_neighbors != -1:
                idx = knn_graph(x, k, batch)[0]
            else:
                # fully connected
                idx = torch.arange(x.shape[0]).repeat(x.shape[0], 1).flatten()
        idx = idx.to(x.device)

        feature = x[idx].view(x.shape[1], idx.shape[0], 1)
        return feature, idx
