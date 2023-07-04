import os

import torch

from src import data
from src.common import decide_total_volume_range, update_reso
from src.encoder import encoder_dict
from src.mo_onet import models, training, generation

# TODO clean this up


def get_model(cfg, device=None, dataset=None, **kwargs):
    """Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    """
    decoder = cfg["model"]["decoder"]
    segmenter = cfg["model"]["segmenter"]
    encoder = cfg["model"]["encoder"]
    n_nodes = cfg["data"]["pointcloud_n"]
    c_dim = cfg["model"]["c_dim"]
    fake_segmentation = cfg["model"]["fake_segmentation"]
    decoder_kwargs = cfg["model"]["decoder_kwargs"]
    segmenter_kwargs = cfg["model"]["segmenter_kwargs"]
    # TODO
    n_classes = 4  # cfg["data"]["n_classes"]
    encoder_kwargs = cfg["model"]["encoder_kwargs"]

    # for pointcloud_crop
    try:
        encoder_kwargs["unit_size"] = cfg["data"]["unit_size"]
        decoder_kwargs["unit_size"] = cfg["data"]["unit_size"]
        segmenter_kwargs["unit_size"] = cfg["data"]["unit_size"]
    except:
        pass
    # local positional encoding
    if "local_coord" in cfg["model"].keys():
        encoder_kwargs["local_coord"] = cfg["model"]["local_coord"]
        decoder_kwargs["local_coord"] = cfg["model"]["local_coord"]
        segmenter_kwargs["local_coord"] = cfg["data"]["local_coord"]
    if "pos_encoding" in cfg["model"]:
        encoder_kwargs["pos_encoding"] = cfg["model"]["pos_encoding"]
        decoder_kwargs["pos_encoding"] = cfg["model"]["pos_encoding"]
        segmenter_kwargs["pos_encoding"] = cfg["data"]["pos_encoding"]

    # update the feature volume/plane resolution
    if cfg["data"]["input_type"] == "pointcloud_crop":
        fea_type = cfg["model"]["encoder_kwargs"]["plane_type"]
        if (dataset.split == "train") or (cfg["generation"]["sliding_window"]):
            recep_field = 2 ** (
                cfg["model"]["encoder_kwargs"]["unet3d_kwargs"]["num_levels"] + 2
            )
            reso = cfg["data"]["query_vol_size"] + recep_field - 1
            if "grid" in fea_type:
                encoder_kwargs["grid_resolution"] = update_reso(reso, dataset.depth)
            if bool(set(fea_type) & set(["xz", "xy", "yz"])):
                encoder_kwargs["plane_resolution"] = update_reso(reso, dataset.depth)
        # if dataset.split == "val": #TODO run validation in room level during training
        else:
            if "grid" in fea_type:
                encoder_kwargs["grid_resolution"] = dataset.total_reso
            if bool(set(fea_type) & set(["xz", "xy", "yz"])):
                encoder_kwargs["plane_resolution"] = dataset.total_reso

    # TODO (GAL) add additional configs

    encoder = encoder_dict[encoder](c_dim=c_dim, **encoder_kwargs)
    decoder = models.decoder_dict[decoder](c_dim=c_dim + 8, **decoder_kwargs)

    segmenter = models.segmenter_dict[segmenter](
        n_points=n_nodes, n_classes=n_classes, **segmenter_kwargs
    )
    # load pretrained segmentation net
    if "pretrained" in segmenter_kwargs:
        ckp = torch.load(
            open(os.path.join(os.getcwd(), segmenter_kwargs["pretrained"]), "rb")
        )
        ckp = segmenter.filter_state_dict(ckp)
        segmenter.load_state_dict(ckp["model_state_dict"], strict=False)

    model = models.MultiObjectONet(
        decoder, segmenter, encoder, fake_segmentation=fake_segmentation, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    """Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    """
    threshold = cfg["test"]["threshold"]
    out_dir = cfg["training"]["out_dir"]
    vis_dir = os.path.join(out_dir, "vis")
    input_type = cfg["data"]["input_type"]

    trainer = training.Trainer(
        model,
        optimizer,
        device=device,
        input_type=input_type,
        vis_dir=vis_dir,
        threshold=threshold,
        weighted_loss=cfg["training"]["weighted_loss"],
        eval_sample=cfg["training"]["eval_sample"],
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    """Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    """

    generator = generation.MultiObjectGenerator3D(
        model,
        device=device,
        threshold=cfg["test"]["threshold"],
        multimesh=cfg["generation"]["multimesh"],
        resolution0=cfg["generation"]["resolution_0"],
        upsampling_steps=cfg["generation"]["upsampling_steps"],
        sample=cfg["generation"]["use_sampling"],
        refinement_step=cfg["generation"]["refinement_step"],
        simplify_nfaces=cfg["generation"]["simplify_nfaces"],
        input_type=cfg["data"]["input_type"],
        padding=cfg["data"]["padding"],
    )
    return generator


def get_data_fields(mode, cfg):
    """Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    points_transform = data.SubsamplePoints(cfg["data"]["points_subsample"])

    input_type = cfg["data"]["input_type"]
    fields = {}
    if cfg["data"]["points_file"] is not None:
        if input_type != "pointcloud_crop":
            fields["points"] = data.PointsField(
                cfg["data"]["points_file"],
                points_transform,
                unpackbits=cfg["data"]["points_unpackbits"],
                multi_files=cfg["data"]["multi_files"],
            )
        else:
            fields["points"] = data.PatchPointsField(
                cfg["data"]["points_file"],
                transform=points_transform,
                unpackbits=cfg["data"]["points_unpackbits"],
                multi_files=cfg["data"]["multi_files"],
            )

    if mode in ("val", "test"):
        points_iou_file = cfg["data"]["points_iou_file"]
        voxels_file = cfg["data"]["voxels_file"]
        if points_iou_file is not None:
            if input_type == "pointcloud_crop":
                fields["points_iou"] = data.PatchPointsField(
                    points_iou_file,
                    unpackbits=cfg["data"]["points_unpackbits"],
                    multi_files=cfg["data"]["multi_files"],
                )
            else:
                fields["points_iou"] = data.PointsField(
                    points_iou_file,
                    unpackbits=cfg["data"]["points_unpackbits"],
                    multi_files=cfg["data"]["multi_files"],
                )
        if voxels_file is not None:
            fields["voxels"] = data.VoxelsField(voxels_file)

    return fields
