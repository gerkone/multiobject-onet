from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import distributions as dist
from sklearn.cluster import DBSCAN
import numpy as np

from src.conv_onet.models import decoder as onetdecoder
from src.mo_onet.models import decoder, segmenter

decoder_dict = {
    "equi_mlp": decoder.EquivariantMLP,
    "simple_local": onetdecoder.LocalDecoder,
    "simple_local_crop": onetdecoder.PatchLocalDecoder,
    "simple_local_point": onetdecoder.LocalPointDecoder,
}

segmenter_dict = {
    "pointnet++": segmenter.PointNet2Segmenter,
}


class MultiObjectONet(ABC, nn.Module):
    """Abstract base class for multi-object occupancy networks."""

    def __init__(self):
        super().__init__()

        self.scene_builder_metadata = {"normalization": None, "baricenters": None}

    @abstractmethod
    def encode_and_segment(self, pc):
        """Encodes and segments the input.

        Args:
            pc (tensor): the input point cloud (batch_sie, n_points, 3)
        Returns:
            codes (list): list of latent conditioned codes
            node_tag (tensor): segmentation node probs (batch_size, n_points, n_classes)
        """
        raise NotImplementedError

    @abstractmethod
    def decode_multi_object(self, q, codes, **kwargs):
        """Returns full scene occupancy probabilities for the sampled points.

        Args:
            q (tensor): points (batch_size, n_sample_points, 3)
            c (tensor): latent conditioned code c
        Returns:
            p_r (tensor): occupancy probs (batch_size, n_objects, n_sample_points)
        """
        raise NotImplementedError

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


class TwoStepMultiObjectONet(MultiObjectONet):
    """Two step (segment first, embed later) multi-object occupancy network.

    Args:
        decoder (nn.Module): decoder network
        segmenter (nn.Module): segmentation network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder, segmenter, encoder=None, device=None):
        super().__init__()

        self.decoder = decoder.to(device)

        self.segmenter = segmenter.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

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
        segmented_objects, node_tags = self.segment_to_single_graphs(pc)
        return self.encode_multi_object(segmented_objects), node_tags

    def encode_multi_object(self, segmented_objects):
        """Encodes the input.

        Args:
            segmented_objects (tensor): list of single object point clouds
        """
        # TODO can be done in parallel
        return [self.encoder(obj) for obj in segmented_objects]

    def segment_to_single_graphs(self, pc):
        """Segments the input point cloud into single shapes.

        Args:
            pc (tensor): the input point cloud (batch_sie, n_points, 3)
        Returns:
            segmented_objects (list): list of single object point clouds
            node_tag (tensor): segmentation node probs (batch_size, n_points, n_classes)
        """
        node_tags, _ = self.segmenter(pc)
        # TODO get scene builder metadata from node_tag and pc
        self.scene_builder_metadata = None
        # split graphs on node_tag
        segmented_objects = self._split_graphs(pc, torch.argmax(node_tags, dim=-1))
        return segmented_objects, node_tags

    def decode_multi_object(self, q, codes, **kwargs):
        # TODO can be done in parallel
        # TODO should operate everywhere in unit cube right?
        return [self.decode_single_object(q, c, **kwargs) for c in codes]

    def decode_single_object(self, q, c, **kwargs):
        logits = self.decoder(q, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def _split_graphs(self, pc, node_tag):
        """Splits the input point cloud into multiple graphs depending on the node tag.

        Args:
            pc (tensor): input point cloud (batch_size, n_points, 3)
            node_tag (tensor): node integer tag (batch_size, n_points)
        Returns:
            graphs (list): list of graphs for each single object
        """
        pc = pc.detach().cpu().numpy()
        node_tag = node_tag.detach().cpu().numpy()
        graphs = []
        tag_range = range(node_tag.min(), node_tag.max() + 1)
        for tag in tag_range:
            # first split by tag directly
            tag_mask = node_tag == tag
            n_nodes = tag_mask.sum(-1)
            graphs.append((torch.tensor(pc[tag_mask]), torch.tensor(n_nodes)))
            # TODO then split each object by grouping close points
            # avg_dist = np.sqrt(np.sum(np.square(pc.reshape(-1, 3) - pc.reshape(-1, 3).mean(0)), axis=-1)).mean()
            # dbscan = DBSCAN(0.5 * avg_dist, min_samples=int(n_nodes.mean() // 5))
            # object_masks = [dbscan.fit_predict(pc_[mask_]) for pc_, mask_ in zip(pc, tag_mask)]

        return graphs

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model


class E2EMultiObjectONet(MultiObjectONet):
    """End-to-end (segment and embed) multi-object occupancy network.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.scene_builder_metadata = {"normalization": None, "baricenters": None}

    def forward(self, q, pc, **kwargs):
        """Performs a forward pass through the network.

        Args:
            q (tensor): sampled points (batch_size, n_sample_points, 3)
            pc (tensor): conditioning input (batch_sie, n_points, 3)
        """
        codes = self.encode_and_segment(pc)
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
        # embed to latent code and segment
        node_embedding, node_tag = self.encoder(pc)
        # TODO get scene builder metadata from node_tag and pc
        self.scene_builder_metadata = None

        return self.pool_codes_segmented(node_embedding, node_tag)

    def decode_multi_object(self, q, codes, **kwargs):
        # TODO should operate everywhere in unit cube right?
        occupancy_probs = []
        for c in codes:
            occupancy_probs.append(self.decode_single_object(q, c, **kwargs))
        return occupancy_probs

    def decode_single_object(self, q, c, **kwargs):
        logits = self.decoder(q, c, **kwargs)
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
