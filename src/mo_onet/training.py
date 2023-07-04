import os
from typing import List

import torch
from torch.nn import functional as F

from src.common import add_key, compute_iou, make_3d_grid
from src.training import BaseTrainer


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
        weighted_loss=False,
        eval_sample=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.weighted_loss = weighted_loss
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, batch):
        """Performs a training step.

        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()

        try:
            if isinstance(batch, List):
                loss = 0.0
                for sample_i in batch:
                    loss += self.compute_loss(sample_i) / len(batch)
                actual_bs = sum(b["points"].shape[0] for b in batch)
            else:
                loss = self.compute_loss(batch)
                actual_bs = batch["points"].shape[0]

            # TODO (GAL) vmap is very slow right now
            # loss = torch.vmap(
            #     self.compute_loss,
            #     in_dims=(0, None),
            #     randomness="same"
            # )(data, n_obj).mean()

            loss.backward()
            self.optimizer.step()

            return loss.item(), actual_bs
        except Exception as e:
            print(f"ERROR (training loop) {e.__doc__}: {e}")
            return None, None

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
        node_tag = data.get("inputs.node_tags").to(device)  # (bs, pc)

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
            # p_out = self.model(points_iou, inputs, **kwargs)
            c, obj_batch = self.model.encode_multi_object(inputs, node_tag)
            p_out = self.model.decode_multi_object(points_iou, c, node_tag=obj_batch)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]
        if threshold not in thresholds:
            thresholds.append(threshold)
        for th in thresholds:
            occ_iou_hat_np = (p_out.probs >= th).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
            if th == threshold:
                eval_dict["iou"] = iou
            else:
                eval_dict[f"iou{int(th * 100)}%"] = iou

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
        target_occ = data.get("points.occ").to(device)  # (bs, n_points,)

        inputs = data.get("inputs", torch.empty(p.size(0), 0)).to(device)
        node_tag = data.get("inputs.node_tags").to(device)  # (bs, pc)

        if "pointcloud_crop" in data.keys():
            # add pre-computed index
            inputs = add_key(
                inputs, data.get("inputs.ind"), "points", "index", device=device
            )
            inputs["mask"] = data.get("inputs.mask").to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get("points.normalized"), "p", "p_n", device=device)

        # segment and split objects
        # node_tag, _ = self.model.segment_to_single_graphs(inputs)

        codes, obj_batch = self.model.encode_multi_object(inputs, node_tag)

        pred_occ = self.model.decode_multi_object(
            p, codes, node_tag=obj_batch
        )  # (bs, n_obj, total_n_points)

        if self.weighted_loss:
            ones_ratio = target_occ.sum() / target_occ.numel()
            weight = torch.where(target_occ > 0, 1 - ones_ratio, ones_ratio + 1e-2)
        else:
            weight = torch.ones_like(target_occ)

        scene_reconstruction_loss = F.binary_cross_entropy_with_logits(
            pred_occ.sum(1), target_occ, reduction="none", weight=weight
        ).sum(-1)

        # average over batch
        total_loss = scene_reconstruction_loss.mean()

        return total_loss
