import torch
import torch.nn as nn
from torch import distributions as dist

from src.conv_onet.models import decoder as onetdecoder
from src.mo_onet.models import decoder, segmenter

decoder_dict = {
    "equi_mlp": decoder.E3Decoder,
    "simple_local": onetdecoder.LocalDecoder,
    "simple_local_crop": onetdecoder.PatchLocalDecoder,
    "simple_local_point": onetdecoder.LocalPointDecoder,
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
        segmented_objects, scene_metadata = self.segment_to_single_graphs(pc)
        return self.encode_multi_object(segmented_objects), scene_metadata

    def encode_multi_object(self, segmented_objects):
        """Encodes the input.

        Args:
            segmented_objects (tensor): list of single object point clouds
        """
        # TODO (GAL) batched
        return [self.encoder(obj) for obj in segmented_objects]

    def segment_to_single_graphs(self, pc):
        """Segments the input point cloud into single shapes.

        Args:
            pc (tensor): the input point cloud (n_points, 3)
        Returns:
            segmented_objects (list): list of single object point clouds
            scene_metadata (dict): scene metadata for scene building
        """
        node_tags, _ = self.segmenter(pc)
        scene_metadata = self.build_scene_metadata(node_tags)
        # split graphs on node_tag
        segmented_objects = self._split_object_instances(pc, node_tags)
        return segmented_objects, scene_metadata

    def decode_multi_object(self, q, codes, logits=True, **kwargs):
        """Returns full scene occupancy probabilities for the sampled points.

        Args:
            q (tensor): (list of object-wise) points (n_sample_points, 3)
            codes (tensor): (list of object-wise) latent object code c
        Returns:
            p_r (list): list of occupancy probs
        """
        # TODO (GAL) batched
        # TODO should operate everywhere in unit cube right?
        assert len(q) == len(codes)
        n_obj = len(q)
        return [self.decode_single_object(q[i], codes[i], logits, **kwargs) for i in range(n_obj)]

    def decode_single_object(self, q, c, logits=True, **kwargs):
        """Returns single object occupancy probabilities for the sampled points.

        Args:
            p (tensor): points (n_sample_points, 3)
            c (tensor): latent conditioned code c
        Returns:
            p_r (tensor): occupancy probs (n_sample_points,)
        """
        if logits:
            return self.decoder(q, c, **kwargs)
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

    def _split_object_instances(self, pc, node_tag):
        """Splits the input point cloud into multiple graphs depending on the node tag.

        Args:
            pc (tensor): input point cloud (n_points, 3)
            node_tag (tensor): node integer tag (n_points,)
        Returns:
            graphs (list): list of graphs for each single object
        """
        graphs = []
        # TODO how is the node tag encoded?
        for tag in node_tag.unique():
            tag_mask = node_tag == tag
            n_nodes = tag_mask.sum(-1)
            graphs.append((pc[tag_mask], n_nodes))
        return graphs
