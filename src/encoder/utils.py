import torch


def knn_graph(x: torch.Tensor, k: int, batch: torch.Tensor) -> torch.Tensor:
    """Naive k-nearest neighbor graph for a set of points.

    Args:
        x (torch.Tensor): Input points (num_points, 3)
        batch (torch.Tensor): Tensor assigning each point to a batch (num_points,)
        k (int): Number of neighbors

    Returns:
        torch.Tensor: Edge indices with shape (2, num_points * k)
    """
    num_points = x.shape[0]
    # naive: compute all pairwise distances
    d = torch.cdist(x, x)
    # exclude self from neighbors
    d = d.fill_diagonal_(torch.inf)
    # exclude points outside batch
    d = torch.where(batch[:, None] == batch[None, :], d, torch.inf)
    _, src = torch.topk(d, k, dim=1, largest=False)
    src = src.reshape(-1)
    dst = torch.arange(num_points, device=x.device).repeat_interleave(k)
    return torch.stack([src, dst], dim=0)


def scatter(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    reduce: str = "sum",
):
    # reduce in ["sum", "prod", "mean", "max", "min"]
    data = data.squeeze()
    segment_ids = segment_ids.flatten()
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    return result.scatter_reduce(0, segment_ids, data, reduce)
