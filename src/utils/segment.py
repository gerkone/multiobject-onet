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

    segmented_pts = []
    segmented_semantics = []

    segmented_pts.append(walls)  # 1st segment is for the walls
    segmented_semantics.append(-1)
    
    for bbox in bboxes:
        bottom_left_x = bbox[0][0]
        bottom_left_z = bbox[0][1]
        up_right_x = bbox[1][0]
        up_right_z = bbox[1][1]
        
        mask_x = (no_walls[:, 0] >= bottom_left_x - eps) & (no_walls[:, 0] <= up_right_x + eps)
        mask_z = (no_walls[:, 2] >= bottom_left_z - eps) & (no_walls[:, 2] <= up_right_z + eps)
        mask = mask_x & mask_z
        seg = no_walls[mask]
        if seg.shape[0] > 0: # there are cases when no object points are in the pointcloud - basicallly < n_objects argets
            segmented_pts.append(seg)

            idx = np.where(np.all(points == seg[0], axis=1))[0][0]  # id of the point in the original pointcloud
            segmented_semantics.append(semantics[idx])  # adding the semantic id
    
    min_seg_pts = min([seg_pts.shape[0] for seg_pts in segmented_pts])
    
    segmented_objs = []
    for obj in segmented_pts:  # batch with objects of different sizes doesn't work - TODO figure out how
        idx = np.round(np.linspace(0, obj.shape[0] - 1, min_seg_pts)).astype(int)
        # np.random.choice(obj, size=min_seg_pts, replace=False) # or random maybe
        subsampled = obj[idx]
        segmented_objs.append(subsampled)

    return segmented_pts






