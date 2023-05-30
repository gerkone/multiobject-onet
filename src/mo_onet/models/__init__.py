import torch
import torch.nn as nn
from torch import distributions as dist
from torch.distributions.utils import probs_to_logits, logits_to_probs


from src.mo_onet.models import decoder, segmenter

decoder_dict = {
    "e3_decoder": decoder.E3Decoder,
    "cbn_decoder": decoder.DecoderCBN,
}

segmenter_dict = {
    "segmenter": segmenter.Segmenter,
}


class MultiObjectONet(nn.Module):
    """Two step (segment first, embed later) multi-object occupancy network.

    Args:
        decoder (nn.Module): decoder network
        segmenter (nn.Module): segmentation network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder, segmenter, encoder, device=None):
        super().__init__()

        self.decoder = decoder.to(device)
        self.segmenter = segmenter.to(device)
        self.encoder = encoder.to(device)

        self._device = device

    def forward(self, q, pc, **kwargs):
        """Performs a forward pass through the network.

        Args:
            q (tensor): sampled points (n_sample_points, 3)
            pc (tensor): conditioning input (n_points, 3)

        Returns:
            p_r (tensor): predicted occupancy values (n_sample_points)
            scene_metadata (dict): scene metadata for scene building
        """
        codes, scene_metadata = self.encode_and_segment(pc)
        p_r = self.decode_multi_object(q, codes, **kwargs)
        return p_r, scene_metadata

    def encode_and_segment(self, pc):
        """Encodes and segments the input.

        Args:
            pc (tensor): the input point cloud (n_points, 3)
        Returns:
            codes (list): list of latent conditioned codes
            scene_metadata (dict): scene metadata for scene building
        """
        node_tag, scene_metadata = self.segment_to_single_graphs(pc)
        return self.encode_multi_object(pc, node_tag), scene_metadata

    def encode_multi_object(self, pc, node_tag):
        """Encodes the input.

        Args:
            pc (tensor): the input point cloud (bs, n_points, 3)
            node_tag (tensor): the node-wise instance tag (bs, n_points)
        """
        bs = pc.shape[0]
        # replace node tags with object indices
        obj_node_tag = torch.zeros_like(node_tag)
        for b in range(bs):
            for i, tag in enumerate(node_tag[b].unique()):
                obj_node_tag[b][node_tag[b] == tag] = i
        n_obj = obj_node_tag.max() + 1
        # add batch offset to obj_node_tag
        batch_offset = torch.arange(0, bs, device=pc.device)[:, None] * n_obj
        obj_node_tag = obj_node_tag + batch_offset
        codes = self.encoder(pc, obj_node_tag)  # (bs, n_obj, c_dim)
        return codes

    def segment_to_single_graphs(self, pc):
        """Segments the input point cloud into single shapes.

        Args:
            pc (tensor): the input point cloud (n_points, 3)
        Returns:
            node_tag (tensor): the node-wise instance tag (n_points,)
            scene_metadata (dict): scene metadata for scene building
        """
        node_tags, _ = self.segmenter(pc)
        scene_metadata = self.build_scene_metadata(node_tags)
        return node_tags, scene_metadata

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
        total_probs = torch.sum(probs, dim=1)  # (bs, n_sample_points,)
        # normalize
        total_probs = (total_probs - total_probs.min()) / (
            total_probs.max() - total_probs.min()
        )
        total_logits = probs_to_logits(total_probs, is_binary=True)
        total_logits = torch.sum(obj_logits, dim=1)
        return dist.Bernoulli(logits=total_logits)

    def build_scene_metadata(self, node_tag):
        """Builds scene metadata for scene building.

        Args:
            node_tag (tensor): node integer tag (n_points,)

        Returns:
            scene_metadata (dict): a Dict containing number of objects, barycenters
                positions and normalization params.
        """
        # TODO get scene builder metadata from node_tag and pc
        return {
            "n_objects": None,
            "barycenters": None,
            "normalization_params": None,
        }
