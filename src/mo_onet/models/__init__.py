import torch
import torch.nn as nn
from torch import distributions as dist

from src.conv_onet.models import decoder as onet_decoder
from src.mo_onet.models import decoder, segmenter

decoder_dict = {
    "equi_mlp": decoder.EquivariantMLP,
    "simple_local": onet_decoder.LocalDecoder,
    "simple_local_crop": onet_decoder.PatchLocalDecoder,
    "simple_local_point": onet_decoder.LocalPointDecoder,
}

segmenter_dict = {
    "pointnet": segmenter.PointNetSegmenter,
}


class TwoStepMultiObjectONet(nn.Module):
    """Two step (segment first, embed later) multi-object occupancy network.

    Args:
        decoder (nn.Module): decoder network
        segmenter (nn.Module): segmentation network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder, segmenter, encoder=None, device=None):
        super().__init__()

        self._decoder = decoder.to(device)

        self._segmenter = segmenter.to(device)

        if encoder is not None:
            self._encoder = encoder.to(device)
        else:
            self._encoder = None

        self._device = device

        self.scene_builder_metadata = {"normalization": None, "baricenters": None}

    def forward(self, q, pc, **kwargs):
        """Performs a forward pass through the network.

        Args:
            q (tensor): sampled points (batch_size, n_sample_points, 3)
            pc (tensor): conditioning input (batch_sie, n_points, 3)
        """
        segmented_objects = self.segment_to_single_graphs(pc)
        codes = self.encode_multi_object(segmented_objects)
        p_r = self.decode_multi_object(q, codes, **kwargs)
        return p_r

    def encode_multi_object(self, segmented_objects):
        """Encodes the input.

        Args:
            segmented_objects (tensor): list of single object point clouds
        """
        # TODO can be done in parallel
        return [self._encoder(obj) for obj in segmented_objects]

    def segment_to_single_graphs(self, pc):
        """Segments the input point cloud into single shapes.

        Args:
            pc (tensor): the input point cloud (batch_sie, n_points, 3)
        """
        node_tags = self._segmenter(pc)
        # TODO get scene builder metadata from node_tag and pc
        self.scene_builder_metadata = None
        # TODO split graphs on node_tag
        segmented_objects = None
        return segmented_objects

    def decode_multi_object(self, q, codes, **kwargs):
        """Returns full scene occupancy probabilities for the sampled points.

        Args:
            q (tensor): points (batch_size, n_sample_points, 3)
            c (tensor): latent conditioned code c
        """
        # TODO should operate everywhere in unit cube right?
        occupancy_probs = []
        for c in codes:
            p_r = self.decode_single_object(q, c, **kwargs)
        return p_r

    def decode_single_object(self, q, c, **kwargs):
        """Returns single object occupancy probabilities for the sampled points.

        Args:
            p (tensor): points (batch_size, n_sample_points, 3)
            c (tensor): latent conditioned code c
        """
        logits = self._decoder(q, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model


class E2EMultiObjectONet(nn.Module):
    """End-to-end (segment and embed) multi-object occupancy network.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self._decoder = decoder.to(device)

        if encoder is not None:
            self._encoder = encoder.to(device)
        else:
            self._encoder = None

        self._device = device

        self.scene_builder_metadata = {"normalization": None, "baricenters": None}

    def forward(self, q, pc, **kwargs):
        """Performs a forward pass through the network.

        Args:
            q (tensor): sampled points (batch_size, n_sample_points, 3)
            pc (tensor): conditioning input (batch_sie, n_points, 3)
        """
        node_embedding, node_tag = self.encode_and_segment(pc)
        codes = self.pool_codes_segmented(node_embedding, node_tag)
        p_r = self.decode_multi_object(q, codes, **kwargs)
        return p_r

    def pool_codes_segmented(self, node_embedding, node_tag):
        """Pool the codes separately for each sub graph.

        Args:
            node_embedding (tensor): multi objects nodes in embedding space
            node_tag (tensor): node tags for segmentation
        """
        # TODO split node embeddings and pool based on node_tag
        codes = None
        return codes

    def encode_and_segment(self, pc):
        """Encodes and segments the input.

        Args:
            pc (tensor): the input point cloud (batch_sie, n_points, 3)
        """
        # TODO what is pc?
        # TODO feature extraction for pc
        node_embedding = None  # self._encoder(pc)
        # TODO segment based on code
        node_tag = None
        # TODO get scene builder metadata from node_tag and pc
        self.scene_builder_metadata = None
        return node_embedding, node_tag

    def decode_multi_object(self, q, codes, **kwargs):
        """Returns full scene occupancy probabilities for the sampled points.

        Args:
            p (tensor): points (batch_size, n_sample_points, 3)
            c (tensor): latent conditioned code c
        """
        # TODO should operate everywhere in unit cube right?
        occupancy_probs = []
        for c in codes:
            occupancy_probs.append(self.decode_single_object(q, c, **kwargs))
        return occupancy_probs

    def decode_single_object(self, q, c, **kwargs):
        """Returns single object occupancy probabilities for the sampled points.

        Args:
            p (tensor): points (batch_size, n_sample_points, 3)
            c (tensor): latent conditioned code c
        """
        logits = self._decoder(q, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model
