import numpy as np


def get_bboxes(bbox_array, xz_groundplane_range):
    x0_plane = type(xz_groundplane_range[1]) is np.ndarray

    xz_bboxes = []
    for bbox in bbox_array:
        xz_bbox = []
        for box in bbox:
            if x0_plane:
                xz_bbox.append([box[0], box[1][0]])
            else:
                xz_bbox.append([box[0][0], box[1]])
        xz_bboxes.append(xz_bbox)

    return np.array(xz_bboxes)


def segment_objects(points, semantics, bboxes, eps=0.007):
    mask = semantics == -1
    walls = points[mask]
    no_walls = points[~mask]
    no_walls_i = np.squeeze(np.argwhere(semantics != -1))
    sem_i = semantics[~mask]

    segmented_pts = []
    segmented_semantics = []
    bboxes_3d = []

    segmented_pts.append(walls)  # 1st segment is for the walls
    segmented_semantics.append(-1)

    for bbox_idx, bbox in enumerate(bboxes):
        bottom_left_x = bbox[0][0]
        bottom_left_z = bbox[0][1]
        up_right_x = bbox[1][0]
        up_right_z = bbox[1][1]

        mask_x = (no_walls[:, 0] >= bottom_left_x - eps) & (
            no_walls[:, 0] <= up_right_x + eps
        )
        mask_z = (no_walls[:, 2] >= bottom_left_z - eps) & (
            no_walls[:, 2] <= up_right_z + eps
        )
        mask = mask_x & mask_z
        seg = no_walls[mask]

        if (
            seg.shape[0] > 0
        ):  # there are cases when no object points are in the pointcloud - basicallly < n_objects argets
            min_coordinates = np.min(seg, axis=0)
            max_coordinates = np.max(seg, axis=0)

            sem_i[mask] = bbox_idx + sem_i[mask] * 10
            bboxes_3d.append((min_coordinates, max_coordinates))
    semantics[no_walls_i] = sem_i

    return semantics, bboxes_3d


def separate_occ(points, occupancy, bboxes3d, eps=0.007, N=4):
    seg_occs = np.tile(np.expand_dims(occupancy, 0), (N, 1))
    for i, bbox in enumerate(bboxes3d):
        mask_x = (points[:, 0] >= bbox[0][0] - eps) & (points[:, 0] <= bbox[1][0] + eps)
        mask_y = (points[:, 1] >= bbox[0][1] - eps) & (points[:, 0] <= bbox[1][1] + eps)
        mask_z = (points[:, 2] >= bbox[0][2] - eps) & (points[:, 0] <= bbox[1][2] + eps)
        mask = mask_x & mask_y & mask_z
        seg_occs[i][~mask] = 0
    return np.stack(seg_occs, axis=0)