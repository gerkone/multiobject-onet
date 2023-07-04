import argparse
import datetime
import os
import time
import shutil
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import trimesh

from src import config, data
from src.checkpoints import CheckpointIO

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Generate multi-object meshes.")
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--no-cuda", action="store_true", help="Do not use cuda.")

    args = parser.parse_args()
    cfg = config.load_config(args.config, "configs/default.yaml")

    is_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if is_cuda else "cpu")

    # Shorthands
    out_dir = cfg["training"]["out_dir"]
    batch_size = cfg["training"]["batch_size"]
    sequential_batches = cfg["training"]["sequential_batches"]
    backup_every = cfg["training"]["backup_every"]

    model_selection_metric = cfg["training"]["model_selection_metric"]
    if cfg["training"]["model_selection_mode"] == "maximize":
        model_selection_sign = 1
    elif cfg["training"]["model_selection_mode"] == "minimize":
        model_selection_sign = -1
    else:
        raise ValueError("model_selection_mode must be " "either maximize or minimize.")

    # Output directory
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"figs/")
    os.makedirs(png_path, exist_ok=True)

    shutil.copyfile(args.config, os.path.join(out_dir, "config.yaml"))

    vis_dataset = config.get_dataset("val", cfg, return_idx=True)

    vis_loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn,
    )
    model_counter = defaultdict(int)
    data_vis_list = []

    iterator = iter(vis_loader)
    for i in range(len(vis_loader)):
        data_vis = next(iterator)
        idx = data_vis["idx"].item()
        model_dict = vis_dataset.get_model_dict(idx)
        category_id = model_dict.get("category", "n/a")
        category_name = vis_dataset.metadata[category_id].get("name", "n/a")
        category_name = category_name.split(",")[0]
        if category_name == "n/a":
            category_name = category_id

        c_it = model_counter[category_id]
        data_vis_list.append({"category": category_name, "it": c_it, "data": data_vis})

        model_counter[category_id] += 1

    # Model
    model = config.get_model(cfg, device=device, dataset=vis_dataset)

    # Generator
    generator = config.get_generator(model, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model)

    try:
        load_dict = checkpoint_io.load("model_best.pt")
    except FileExistsError:
        load_dict = dict()

    metric_val_best = load_dict.get("loss_val_best", -model_selection_sign * np.inf)

    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"Generating from {cfg['method']} with {nparams} parameters.\n"
        f"Best validation metric ({model_selection_metric}): {metric_val_best}"
    )

    base = data_vis_list[8]["data"]
    multiobject = (
        "inputs.node_tags" in data_vis_list[0]["data"]
        and not cfg["model"]["fake_segmentation"]
    )

    if multiobject:
        base_trans = {}
        for id in base["inputs.node_tags"].unique():
            id = id.item()
            same = base["inputs.node_tags"] == id
            base_trans[id] = base["inputs"][same].mean(0)

        mesh, object_meshes = generator.generate_mesh(
            base, return_stats=False, objwise_meshes=True
        )

    for it, dd in enumerate(data_vis_list):
        k = 0
        while True:
            k += 1
            pc = dd["data"]
            if multiobject:
                obj_trans = []
                for id in pc["inputs.node_tags"].unique():
                    id = id.item()
                    same = pc["inputs.node_tags"] == id
                    centroid = pc["inputs"][same].mean(0)
                    trn = trimesh.transformations.translation_matrix(
                        centroid - base_trans[id]
                    )
                    obj_trans.append(trn)

                object_meshes_trn = generator.transform_objects(
                    object_meshes, obj_trans
                )
                mesh = trimesh.util.concatenate(object_meshes_trn)

            else:
                mesh = generator.generate_mesh(pc, return_stats=False)

            scene: trimesh.Scene = mesh.scene()
            rotatex = trimesh.transformations.rotation_matrix(
                angle=np.radians(60), direction=[-1, 0, 0], point=scene.centroid
            )
            rotatey = trimesh.transformations.rotation_matrix(
                angle=np.radians(20), direction=[0, 1, 1], point=scene.centroid
            )
            rotate = np.dot(rotatex, rotatey)
            for i in range(1):
                camera_old, _ = scene.graph[scene.camera.name]
                camera_new = np.dot(rotate, camera_old)

                scene.graph[scene.camera.name] = camera_new

                try:
                    file_name = os.path.join(png_path, f"render_{k}.png")
                    print(f"Saving image {file_name}")

                    png = scene.save_image(resolution=[800, 800], visible=True)
                    with open(file_name, "wb") as f:
                        f.write(png)
                        f.close()
                except Exception as e:
                    print(f"Error {e.__doc__} {e}")

            if k == 60:
                break
