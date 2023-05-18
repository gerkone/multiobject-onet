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

        self.scene_builder_metadata = {
            "n_objects": None,
            "barycenters": None,
            "normalization_params": None,
        }

        self._device = device

    def forward(self, q, pc, **kwargs):
        """Performs a forward pass through the network.

        Args:
            q (tensor): sampled points (batch_size, n_sample_points, 3)
            pc (tensor): conditioning input (batch_sie, n_points, 3)
        """
        codes, _ = self.encode_and_segment(pc)
        p_r = self.decode_multi_object(q, codes, **kwargs)
        return p_r

    def encode_and_segment(self, pc):
        """Encodes and segments the input.

        Args:
            pc (tensor): the input point cloud (batch_sie, n_points, 3)
        Returns:
            codes (list): list of latent conditioned codes
        """
        segmented_objects = self.segment_to_single_graphs(pc)
        return self.encode_multi_object(segmented_objects)

    def encode_multi_object(self, segmented_objects):
        """Encodes the input.

        Args:
            segmented_objects (tensor): list of single object point clouds
        """
        # TODO vmap
        return [self.encoder(obj) for obj in segmented_objects]

    def segment_to_single_graphs(self, pc):
        """Segments the input point cloud into single shapes.

        Args:
            pc (tensor): the input point cloud (batch_sie, n_points, 3)
        Returns:
            segmented_objects (list): list of single object point clouds
        """
        node_tags, _ = self.segmenter(pc)
        # TODO get scene builder metadata from node_tag and pc
        self.scene_builder_metadata = None
        # split graphs on node_tag
        segmented_objects = self._split_object_instances(pc, node_tags)
        return segmented_objects

    def decode_multi_object(self, q, codes, **kwargs):
        """Returns full scene occupancy probabilities for the sampled points.

        Args:
            q (tensor): points (batch_size, n_sample_points, 3)
            c (tensor): latent conditioned code c
        Returns:
            p_r (list): list of occupancy probs
        """
        # TODO vmap
        # TODO should operate everywhere in unit cube right?
        return [self.decode_single_object(q, c, **kwargs) for c in codes]

    def decode_single_object(self, q, c, **kwargs):
        """Returns single object occupancy probabilities for the sampled points.

        Args:
            p (tensor): points (batch_size, n_sample_points, 3)
            c (tensor): latent conditioned code c
        Returns:
            p_r (tensor): occupancy probs (batch_size, n_sample_points)
        """
        logits = self.decoder(q, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def _split_object_instances(self, pc, node_tag):
        """Splits the input point cloud into multiple graphs depending on the node tag.

        Args:
            pc (tensor): input point cloud (batch_size, n_points, 3)
            node_tag (tensor): node integer tag (batch_size, n_points)
        Returns:
            graphs (list): list of graphs for each single object
        """
        graphs = []
        for tag in range(node_tag.min(), node_tag.max() + 1):
            tag_mask = node_tag == tag
            n_nodes = tag_mask.sum(-1)
            graphs.append((pc[tag_mask], n_nodes))
        return graphs
