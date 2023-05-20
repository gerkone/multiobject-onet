import torch


def crop_occupancy_grid(p, occ, segmented_objects, margin=0.1):
    """Crop the occupancy grid for each object.

    Args:
        p (torch.Tensor): sample points (batch_size, n_points, 3)
        occ (torch.Tensor): occupancy grid (batch_size, n_points)
        segmented_objects (list): list of graphs for each single object
        seg (torch.Tensor): instance segmentation id (batch_size, n_points)

    Returns:
        p_crops (list): list of cropped sample points
        occ_crops (list): list of cropped occupancy grid
    """
    p_crops = []
    occ_crops = []
    idx_crop = []
    l_margin = 1 - margin
    u_margin = 1 + margin
    for graph, _ in segmented_objects:
        crop_idxs = (
            (p[..., 0] >= l_margin * torch.min(graph[:, 0]))
            & (p[..., 0] <= u_margin * torch.max(graph[:, 0]))
            & (p[..., 1] >= l_margin * torch.min(graph[:, 1]))
            & (p[..., 1] <= u_margin * torch.max(graph[:, 1]))
            & (p[..., 2] >= l_margin * torch.min(graph[:, 2]))
            & (p[..., 2] <= u_margin * torch.max(graph[:, 2]))
        )
        p_crops.append(p[crop_idxs, :])
        occ_crops.append(occ[crop_idxs])
        idx_crop.append(crop_idxs)
    # NOTE: reduces the number of sample points as it crops around the segmented objects
    return p_crops, occ_crops, idx_crop
