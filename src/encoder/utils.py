import torch
import warnings


def knn_graph_topk(x: torch.Tensor, k: int, batch: torch.Tensor) -> torch.Tensor:
    """Naive k-nearest neighbor graph for a set of points.

    Args:
        x (torch.Tensor): Input points (num_points, 3)
        batch (torch.Tensor): Tensor assigning each point to a batch (num_points,)
        k (int): Number of neighbors

    Returns:
        torch.Tensor: Edge indices with shape (2, num_points * k)
    """
    warnings.warn("[DEPRECATED] Use knn_graph from torch-cluster instead.")
    num_points = x.shape[0]
    # computes all pairwise distances -> blows up memory for large inputs
    # TODO (GAL) use knn from torch-cluster
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
    """Scatter operation for a batch of segment ids.

    Args:
        data (torch.Tensor): Input data (num_points, channels)
        segment_ids (torch.Tensor): Segment ids (num_points,)
        num_segments (int): Number of segments
        reduce (str, optional): Reduce operation. "sum", "prod", "mean", "amax", "amin"
    """
    assert reduce in ["sum", "prod", "mean", "amax", "amin"]
    data = data.squeeze()
    segment_ids = segment_ids.flatten()
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(1).expand_as(data)
    return result.scatter_reduce(0, segment_ids, data, reduce)
