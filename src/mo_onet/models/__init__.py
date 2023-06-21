import torch
import torch.nn as nn
from torch import distributions as dist
from torch.distributions.utils import logits_to_probs

from src.encoder.gconv import MOGConv
from src.mo_onet.models import decoder, segmenter


decoder_dict = {
    "e3_decoder": decoder.E3Decoder,
    "cbn_decoder": decoder.CBNDecoder,
    "gconv_decoder": decoder.DGCNNDecoder,
}

segmenter_dict = {
    "segmenter": segmenter.Segmenter,
}


class MultiObjectONet(nn.Module):
    """Two step (segment first, embed later) multi-object occupancy network.

    Args:
        decoder (nn.Module): decoder network
        segmenter (nn.Module): segmentation network
        object_encoder (nn.Module): object-wise encoder network
        scene_encoder (nn.Module): scene encoder network
        device (device): torch device
    """

    def __init__(self, decoder, segmenter, object_encoder, device=None):
        super().__init__()

        self.decoder = decoder.to(device)
        self.segmenter = segmenter.to(device)
        self.object_encoder = object_encoder.to(device)
        self.scene_encoder = MOGConv(c_dim=8, hidden_size=16, n_neighbors=-1).to(device)

        self._fake_segmentation = False
        self._device = device

    def forward(self, q, pc, **kwargs):
        """Performs a forward pass through the network.

        Args:
            q (tensor): sampled points (n_sample_points, 3)
            pc (tensor): conditioning input (n_points, 3)

        Returns:
            p_r (tensor): predicted occupancy values (n_sample_points)
        """
        codes, _ = self.encode_and_segment(pc)
        p_r = self.decode_multi_object(q, codes, **kwargs)
        return p_r

    def encode_and_segment(self, pc):
        """Encodes and segments the input.

        Args:
            pc (tensor): the input point cloud (n_points, 3)
        Returns:
            codes (list): list of latent conditioned codes
        """
        node_tag = self.segment_to_single_graphs(pc)
        return self.encode_multi_object(pc, node_tag)

    def encode_multi_object(self, pc, node_tag):
        """Encodes the input.

        Args:
            pc (tensor): the input point cloud (bs, n_points, 3)
            node_tag (tensor): the node-wise instance tag (bs, n_points)
        """
        if self._fake_segmentation:
            node_tag = torch.zeros_like(node_tag)
        bs = node_tag.shape[0]
        obj_node_tag = torch.zeros_like(node_tag, dtype=torch.long)
        for b in range(bs):
            for i, tag in enumerate(node_tag[b].unique()):
                obj_node_tag[b][node_tag[b] == tag] = i
        n_obj = obj_node_tag.max() + 1
        # add batch offset to obj_node_tag
        batch_offset = (torch.arange(0, bs, device=pc.device)[:, None] * n_obj).long()
        obj_batch = obj_node_tag + batch_offset

        # objec-wise encoding
        obj_codes = self.object_encoder(pc, obj_batch)  # (bs, n_obj, obj_c_dim)

        # scene encoding from bounding boxes
        # compute object-wise boundary boxes
        bbox = self._get_bbox(pc, bs, n_obj, obj_batch)  # (bs, n_obj * 2, 3)
        barycenter = bbox.view(bs, 2, n_obj, 3).mean(dim=1)  # (bs, n_obj, 3)
        keypoints = torch.cat([bbox, barycenter], dim=1)  # (bs, n_obj * 3, 3)

        scene_batch = torch.arange(0, bs, device=pc.device)[:, None].repeat(
            1, 3 * n_obj
        )

        scene_code = self.scene_encoder(keypoints, scene_batch)[1].mean(
            dim=1, keepdim=True
        )  # (bs, n_obj, scene_c_dim)

        # full codes = object codes + scene code
        n_nodes = obj_codes[0].shape[1]
        codes = (
            obj_codes[0],
            torch.cat([obj_codes[1], scene_code.repeat(1, n_nodes, 1)], dim=-1),
        )

        return codes, obj_batch

    def segment_to_single_graphs(self, pc):
        """Segments the input point cloud into single shapes.

        Args:
            pc (tensor): the input point cloud (n_points, 3)
        Returns:
            node_tag (tensor): the node-wise instance tag (n_points,)
        """
        node_tags, _ = self.segmenter(pc)
        return node_tags

    def decode_multi_object(self, p, codes, **kwargs):
        """Returns full scene occupancy probabilities for the sampled points.

        Args:
            p (tensor): sample points (n_sample_points, 3)
            codes (tensor): object-wise latent object code (n_obj, c_dim)
        Returns:
            p_r (tensor): scene occupancy probs
        """
        obj_logits = self.decoder(p, codes, **kwargs)  # (bs * n_obj, n_sample_points)
        if self.training:
            # return object-wise logits
            return obj_logits
        # sum over objects in prob space
        probs = logits_to_probs(obj_logits, is_binary=True)
        total_probs = probs
        return dist.Bernoulli(probs=total_probs)

    def _get_bbox(self, pc, bs, n_obj, obj_batch):
        bbox = torch.zeros((bs, 2 * n_obj, 3), device=pc.device)
        for b in range(bs):
            for i in range(0, 2 * n_obj, 2):
                mask = obj_batch[b] == i
                if mask.sum() == 0:
                    continue
                bbox[b, i] = pc[b, mask].min(0)[0]
                bbox[b, i + 1] = pc[b, mask].max(0)[0]
        return bbox
