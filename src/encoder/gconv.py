import torch
import torch.nn as nn

from .utils import knn_graph_topk, scatter


class MOGConv(nn.Module):
    def __init__(self, c_dim=128, hidden_size=128):
        super().__init__()
        self.c_dim = c_dim
        self.k = 20

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
            nn.Conv2d(6, hidden_size, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=False),
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

        self.readout = nn.Conv1d(hidden_size, c_dim, kernel_size=1, bias=False)

    def forward(self, x, node_tag, n_obj):
        """MOGConv forward.

        Args:
            x (torch.Tensor): Point cloud (bs, n_nodes, 3)
            node_tag (torch.Tensor): Node tag index for aggregation (bs, n_nodes)
            n_obj (int): number of objects in the scenes
        """
        bs, n_nodes, _ = x.shape
        x = x.view(-1, 3).squeeze()  # (bs * n_nodes, 3)
        node_tag = node_tag.view(-1)  # (bs * n_nodes,)
        with torch.no_grad():
            n_neighbors = self.k * 18
            idx = knn_graph_topk(x, n_neighbors, node_tag)[0].view(
                bs * n_nodes, n_neighbors
            )  # (bs * n_nodes, k)

        x = x.permute(1, 0)

        x = self._transform(
            x, k=self.k, idx=idx[:, : n_neighbors // 18]
        )  # (3*2, bs * n_nodes, k)
        x = self.conv1(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv2(x)  # (hidden_size, bs * n_nodes, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (hidden_size, bs * n_nodes)

        x = self._transform(
            x1, k=self.k, idx=idx[:, : n_neighbors // 9][:, ::2]
        )  # (hidden_size * 2, bs * n_nodes, k)
        x = self.conv3(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv4(x)  # (hidden_size, bs * n_nodes, k)
        x2 = x.max(dim=-1, keepdim=False)[0] + x1  # (hidden_size, bs * n_nodes)

        x = self._transform(
            x2, k=self.k, idx=idx[:, : n_neighbors // 3][:, ::6]
        )  # (hidden_size * 2, bs * n_nodes, k)
        x = self.conv5(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv6(x)
        x3 = x.max(dim=-1, keepdim=False)[0] + x2  # (hidden_size, bs * n_nodes)

        x = self._transform(
            x3, k=self.k, idx=idx[:, :n_neighbors][:, ::18]
        )  # (hidden_size * 2, bs * n_nodes, k)
        x = self.conv7(x)  # (hidden_size, bs * n_nodes, k)
        x = self.conv8(x)  # (hidden_size, bs * n_nodes, k)
        x4 = x.max(dim=-1, keepdim=False)[0] + x3  # (hidden_size, bs * n_nodes)

        x = torch.cat((x1, x2, x3, x4), dim=0)  # (hidden_size * 4, bs * n_nodes)

        x = self.conv9(x)  # (hidden_size * 2, bs * n_nodes)
        x = x.max(dim=-1, keepdim=True)[0]  # (hidden_size * 2, 1)

        x = x.repeat(1, bs * n_nodes)  # (hidden_size * 2, bs * n_nodes)

        x4 = torch.cat((x, x4), dim=0)  # (hidden_size * 2 + hidden_size, bs * n_nodes)
        x4 = self.conv10(x4)  # (hidden_size * 2, bs * n_nodes)

        x3 = torch.cat((x4, x3), dim=0)  # (hidden_size * 2 + hidden_size, bs * n_nodes)
        x3 = self.conv11(x3)  # (hidden_size * 2, bs * n_nodes)

        x2 = torch.cat((x3, x2), dim=0)  # (hidden_size * 2 + hidden_size, bs * n_nodes)
        x2 = self.conv12(x2)  # (hidden_size, bs * n_nodes)

        x1 = torch.cat((x2, x1), dim=0)  # (hidden_size + hidden_size, bs * n_nodes)
        x1 = self.conv13(x1)  # (hidden_size, bs * n_nodes)

        # readout to global code
        code = self.readout(x1).permute(1, 0)  # (bs * n_nodes, c_dim)
        # collect codes on objects
        codes = scatter(code, node_tag.long(), bs * n_obj, "mean")  # (n_obj, c_dim)
        codes = codes.view(bs, n_obj, -1)  # (bs, n_obj, c_dim)
        return codes

    def _transform(self, x, k=20, batch=None, idx=None, dim9=False, featdiff=True):
        n_nodes = x.shape[1]
        if idx is None:
            if dim9 == False:
                idx = knn_graph(x, k, batch)  # (batch_size, bs * n_nodes, k)
            else:
                idx = knn_graph(x[:, 6:], k, batch)
        idx = idx.contiguous()

        n_dims = x.shape[0]

        x = x.transpose(1, 0).contiguous()
        feature = x[idx]
        x = x.view(n_nodes, 1, n_dims).repeat(1, k, 1)

        if featdiff:
            feature = torch.cat((feature - x, x), dim=2).permute(2, 0, 1).contiguous()
        else:
            feature = torch.cat((feature, x), dim=2).permute(2, 0, 1).contiguous()

        return feature  # (2*n_dims, bs * n_nodes, k)


# TODO (GAL) wip


class MOE3GConv(MOGConv):
    pass
