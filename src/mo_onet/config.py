import os

import torch

from src import data
from src.common import decide_total_volume_range, update_reso
from src.encoder import encoder_dict
from src.mo_onet import generation, models, scene_builder, training

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
    dim = cfg["data"]["dim"]
    n_nodes = cfg["data"]["pointcloud_n"]
    c_dim = cfg["model"]["c_dim"]
    decoder_kwargs = cfg["model"]["decoder_kwargs"]
    segmenter_kwargs = cfg["model"]["segmenter_kwargs"]
    # TODO
    n_classes = 4  # cfg["data"]["n_classes"]
    encoder_kwargs = cfg["model"]["encoder_kwargs"]

    padding = cfg["data"]["padding"]

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

    decoder = models.decoder_dict[decoder](c_dim=c_dim, **decoder_kwargs)
    encoder = encoder_dict[encoder](c_dim=c_dim, **encoder_kwargs)

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

    model = models.MultiObjectONet(decoder, segmenter, encoder, device=device)

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

    if cfg["data"]["input_type"] == "pointcloud_crop":
        # calculate the volume boundary
        query_vol_metric = cfg["data"]["padding"] + 1
        unit_size = cfg["data"]["unit_size"]
        recep_field = 2 ** (
            cfg["model"]["encoder_kwargs"]["unet3d_kwargs"]["num_levels"] + 2
        )
        if "unet" in cfg["model"]["encoder_kwargs"]:
            depth = cfg["model"]["encoder_kwargs"]["unet_kwargs"]["depth"]
        elif "unet3d" in cfg["model"]["encoder_kwargs"]:
            depth = cfg["model"]["encoder_kwargs"]["unet3d_kwargs"]["num_levels"]

        vol_info = decide_total_volume_range(
            query_vol_metric, recep_field, unit_size, depth
        )

        grid_reso = cfg["data"]["query_vol_size"] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg["data"]["query_vol_size"] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg["generation"]["sliding_window"]:
            vol_bound = {
                "query_crop_size": query_vol_size,
                "input_crop_size": input_vol_size,
                "fea_type": cfg["model"]["encoder_kwargs"]["plane_type"],
                "reso": grid_reso,
            }

    else:
        vol_bound = None
        vol_info = None

    generator = generation.MultiObjectGenerator3D(
        model,
        scene_builder=scene_builder.SceneBuilder(model.scene_builder_metadata),
        device=device,
        threshold=cfg["test"]["threshold"],
        resolution0=cfg["generation"]["resolution_0"],
        upsampling_steps=cfg["generation"]["upsampling_steps"],
        sample=cfg["generation"]["use_sampling"],
        refinement_step=cfg["generation"]["refinement_step"],
        simplify_nfaces=cfg["generation"]["simplify_nfaces"],
        input_type=cfg["data"]["input_type"],
        padding=cfg["data"]["padding"],
        vol_info=vol_info,
        vol_bound=vol_bound,
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