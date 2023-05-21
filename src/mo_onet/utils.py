import torch


def crop_occupancy_grid(p, pc, occ, node_tag, n_obj):
    """Crops the occupancy grid for each object (for training).

    Args:
        p (torch.Tensor): point cloud (n_points, 3)
        pc (torch.Tensor): conditioning point cloud (n_nodes, 3)
        occ (torch.Tensor): occupancy grid (n_points,)
        node_tag (torch.Tensor): node tag (n_points,)
        n_obj (int): number of objects

    Returns:
        occ_masked (list): list of cropped occupancy grid
    """
    # TODO (GAL) is this even possible?
    raise NotImplementedError
