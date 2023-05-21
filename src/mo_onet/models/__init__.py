import torch
import torch.nn as nn
from torch import distributions as dist

from src.mo_onet.models import decoder, segmenter

decoder_dict = {
    "e3_decoder": decoder.E3Decoder,
    "cbn_decoder": decoder.DecoderCBatchNorm,
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
        return self.encode_multi_object(node_tag), scene_metadata

    def encode_multi_object(self, pc, node_tag, n_obj=None):
        """Encodes the input.

        Args:
            pc (tensor): the input point cloud (n_points, 3)
            node_tag (tensor): the node-wise instance tag (n_points)
            n_obj (int): number of objects in the scene
        """
        return self.encoder(pc, node_tag, n_obj)

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

    def decode_multi_object(self, q, codes, **kwargs):
        """Returns full scene occupancy probabilities for the sampled points.

        Args:
            q (tensor): (list of object-wise) points (n_sample_points, 3)
            codes (tensor): (list of object-wise) latent object code c
        Returns:
            p_r (list): list of occupancy probs
        """
        assert len(q) == len(codes[0]) == len(codes[1])
        # TODO should operate everywhere in unit cube right?
        logits = self.decoder(q, codes, **kwargs)  # (n_obj, n_sample_points)
        # sum over objects
        logits = torch.sum(logits, dim=0)  # (n_sample_points,)
        if self.training:
            return logits
        return dist.Bernoulli(logits=logits)

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
