import os

import torch
from torch.nn import functional as F

from src.common import add_key, compute_iou, make_3d_grid
from src.mo_onet.utils import crop_occupancy_grid
from src.training import BaseTrainer

# TODO


class Trainer(BaseTrainer):
    """Trainer object for the multi-object Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    """

    def __init__(
        self,
        model,
        optimizer,
        device=None,
        input_type="pointcloud",
        vis_dir=None,
        threshold=0.5,
        eval_sample=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        """Performs a training step.

        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()

        # vmap over batch size
        loss = torch.vmap(self.compute_loss, randomness="same")(data).mean()
        # TODO (GAL) debugging only
        # loss = self.compute_loss({k: v[0] for k, v in data.items()})
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, data):
        """Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get("points").to(device)
        # occ = data.get("points.occ").to(device)

        inputs = data.get("inputs", torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get("voxels")

        points_iou = data.get("points_iou").to(device)
        occ_iou = data.get("points_iou.occ").to(device)

        batch_size = points.size(0)

        kwargs = {}

        # add pre-computed index
        inputs = add_key(
            inputs, data.get("inputs.ind"), "points", "index", device=device
        )
        # add pre-computed normalized coordinates
        points = add_key(
            points, data.get("points.normalized"), "p", "p_n", device=device
        )
        points_iou = add_key(
            points_iou, data.get("points_iou.normalized"), "p", "p_n", device=device
        )

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict["iou"] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, voxels_occ.shape[1:]
            )
            points_voxels = points_voxels.expand(batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(
                    points_voxels, inputs, sample=self.eval_sample, **kwargs
                )

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict["iou_voxels"] = iou_voxels

        return eval_dict

    def compute_loss(self, data):
        """Computes the loss.

        Args:
            data (dict): data dictionary
        """
        device = self.device
        p = data.get("points").to(device)
        occ = data.get("points.occ").to(device)  # (n_points,)
        inputs = data.get("inputs", torch.empty(p.size(0), 0)).to(device)
        # TODO (NINA) where to get seg_target from?
        # seg_target = data.get("seg_target").to(device)
        seg_target = torch.randint(0, 4, (inputs.shape[0],)).to(device)

        if "pointcloud_crop" in data.keys():
            # add pre-computed index
            inputs = add_key(
                inputs, data.get("inputs.ind"), "points", "index", device=device
            )
            inputs["mask"] = data.get("inputs.mask").to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get("points.normalized"), "p", "p_n", device=device)

        # segment and split objects
        # TODO ground truth segmentation
        node_tag = seg_target
        segmented_objects = self.model._split_object_instances(inputs, node_tag)
        # encoder
        codes = self.model.encode_multi_object(segmented_objects)

        kwargs = {}

        # TODO (GAL) find a way around dynamic indexing for cropping occupancy grid
        # p_crop, occ_crop, _ = crop_occupancy_grid(
        #     p, occ, segmented_objects
        # )  # list n_obj (n_points_per_obj,)
        p_crop = [p] * len(codes)
        occ_crop = [occ] * len(codes)

        pred_occ = self.model.decode_multi_object(
            p_crop, codes, logits=True, **kwargs
        )  # list n_obj (n_points_per_obj,)
        pred_occ = torch.cat([out for out in pred_occ])  # (total_n_points,)
        occ_crop = torch.cat(occ_crop)  # (total_n_points,)

        scene_reconstruction_loss = F.binary_cross_entropy_with_logits(
            pred_occ, occ_crop, reduction="none"
        ).mean()

        total_loss = scene_reconstruction_loss

        return total_loss
