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


def segment_objects(instances):
    return instances + 1  # for having nodes 0-8


def separate_occ(instances, N=8):
    tags = np.sort(np.unique(instances))
    seg_occs = np.zeros((N + 1, instances.shape[0]))  # +1 for ground floor
    sems = []
    for i, label in enumerate(tags):
        if label < 8:
            mask = instances == label
            seg_occs[i][mask] = 1.0
            sems.append(label + 1)
    return seg_occs, np.stack(sems, axis=0)
