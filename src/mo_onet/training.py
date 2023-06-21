import os

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

    def train_step(self, data):
        """Performs a training step.

        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()

        try:
            bs = data["points"].shape[0]
            same_n_obj = all(
                data["inputs.node_tags"][i].unique().shape[0]
                == data["inputs.node_tags"][0].unique().shape[0]
                for i in range(bs)
            )

            # filter elements with different number of objects
            mask = [
                data["inputs.node_tags"][i].unique().shape[0] == 5 for i in range(bs)
            ]

            if not any(mask):
                return 0.0, 0

            batch = {k: v[mask] for k, v in data.items()}

            loss = self.compute_loss(batch)

            # TODO (GAL) vmap is very slow right now. Try to go back to batching
            # loss = torch.vmap(self.compute_loss, in_dims=(0, None), randomness="same")(
            #     data, n_obj
            # ).mean()

            loss.backward()
            self.optimizer.step()

            return loss.item(), batch["points"].shape[0]
        except Exception as e:
            print(e)
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
            codes, obj_batch = self.model.encode_multi_object(inputs, node_tag)
            p_out = self.model.decode_multi_object(
                points_iou, codes, node_tag=obj_batch
            )  # (bs, n_obj, total_n_points)

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
        target_occ = data.get("points.occ").to(device)  # (bs, n_points,)
        node_occs = data.get("inputs.node_occs").to(device)  # (bs, n_obj, n_points)

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
        # TODO real instance segmentation
        # node_tag, _ = self.model.segment_to_single_graphs(inputs)

        codes, obj_batch = self.model.encode_multi_object(inputs, node_tag)

        pred_occ = self.model.decode_multi_object(
            p, codes, node_tag=obj_batch
        )  # (bs, n_obj, total_n_points)

        assert pred_occ.shape == node_occs.shape, "Must return object-wise occupancy."

        if self.weighted_loss:
            ones_ratio = target_occ.sum() / target_occ.numel()
            weight = torch.where(target_occ > 0, 1 - ones_ratio, ones_ratio + 1e-3)
        else:
            weight = torch.ones_like(target_occ)

        object_reconstruction_loss = (
            F.binary_cross_entropy_with_logits(
                pred_occ, node_occs, reduce=False
            )
            # sum over points
            .sum(-1)
            # sum over objects
            .sum(-1)
        )

        scene_reconstruction_loss = F.binary_cross_entropy_with_logits(
            pred_occ.sum(1), target_occ, reduction="none", weight=weight
        ).sum(-1)

        # average over batch
        total_loss = (
            scene_reconstruction_loss.mean() + object_reconstruction_loss.mean()
        )

        return total_loss
