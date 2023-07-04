import argparse
import datetime
import os
import time
import shutil
from collections import defaultdict

import warnings

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from src import config, data
from src.checkpoints import CheckpointIO

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Train a 3D reconstruction model.")
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument(
        "--new", action="store_true", help="Ignore checkpoints, start new training."
    )
    parser.add_argument("--test", action="store_true", help="Run test evaulation.")
    parser.add_argument("--no-cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit-after",
        type=int,
        default=-1,
        help="Checkpoint and exit after specified number of seconds"
        "with exit code 2.",
    )

    args = parser.parse_args()
    cfg = config.load_config(args.config, "configs/default.yaml")

    is_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = datetime.datetime.now()

    # Shorthands
    out_dir = cfg["training"]["out_dir"]
    batch_size = cfg["training"]["batch_size"]
    sequential_batches = cfg["training"]["sequential_batches"]
    backup_every = cfg["training"]["backup_every"]
    vis_n_outputs = cfg["generation"]["vis_n_outputs"]
    exit_after = args.exit_after

    model_selection_metric = cfg["training"]["model_selection_metric"]
    if cfg["training"]["model_selection_mode"] == "maximize":
        model_selection_sign = 1
    elif cfg["training"]["model_selection_mode"] == "minimize":
        model_selection_sign = -1
    else:
        raise ValueError("model_selection_mode must be " "either maximize or minimize.")

    # Output directory
    os.makedirs(out_dir, exist_ok=True)

    shutil.copyfile(args.config, os.path.join(out_dir, "config.yaml"))

    # Dataset
    train_dataset = config.get_dataset("train", cfg)
    val_dataset = config.get_dataset("val", cfg, return_idx=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfg["training"]["n_workers_val"],
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn,
    )

    if args.test:
        test_dataset = config.get_dataset("test", cfg, return_idx=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=cfg["training"]["n_workers_val"],
            shuffle=False,
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn,
        )

    # For visualizations
    vis_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn,
    )
    model_counter = defaultdict(int)
    data_vis_list = []

    # Build a data dictionary for visualization
    iterator = iter(vis_loader)
    for i in range(len(vis_loader)):
        data_vis = next(iterator)
        idx = data_vis["idx"].item()
        model_dict = val_dataset.get_model_dict(idx)
        category_id = model_dict.get("category", "n/a")
        category_name = val_dataset.metadata[category_id].get("name", "n/a")
        category_name = category_name.split(",")[0]
        if category_name == "n/a":
            category_name = category_id

        c_it = model_counter[category_id]
        if c_it < vis_n_outputs:
            data_vis_list.append(
                {"category": category_name, "it": c_it, "data": data_vis}
            )

        model_counter[category_id] += 1

    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)

    # Generator
    generator = config.get_generator(model, cfg, device=device)

    # Intialize training
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    if not args.new:
        try:
            load_dict = checkpoint_io.load("model.pt")
        except FileExistsError:
            load_dict = dict()
        epoch_it = load_dict.get("epoch_it", 0)
        it = load_dict.get("it", 0)
        metric_val_best = load_dict.get("loss_val_best", -model_selection_sign * np.inf)
    else:
        epoch_it = 0
        it = 0
        metric_val_best = np.inf

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf
    print(
        f"Current best validation metric ({model_selection_metric}): {metric_val_best}"
    )
    logger = SummaryWriter(os.path.join(out_dir, "logs"))

    # Shorthands
    print_every = cfg["training"]["print_every"]
    checkpoint_every = cfg["training"]["checkpoint_every"]
    validate_every = cfg["training"]["validate_every"]
    visualize_every = cfg["training"]["visualize_every"]

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {nparameters}")

    print(f"output path: {cfg['training']['out_dir']}")

    if not args.test:
        print("Training")
        losses = []
        bss = []
        while True:
            epoch_it += 1
            try:
                train_iter = iter(train_loader)
                for batch in train_iter:
                    it += 1

                    if sequential_batches > 1:
                        batches = [batch]
                        try:
                            for _ in range(sequential_batches - 1):
                                batches.append(next(train_iter))
                        except StopIteration:
                            pass
                    else:
                        batches = batch

                    out = trainer.train_step(batches)
                    if isinstance(out, tuple):
                        loss, actual_bs = out
                    else:
                        loss, actual_bs = out, batch_size

                    if loss is not None:
                        logger.add_scalar("train/loss", loss, it)
                        losses.append(loss)
                    if actual_bs is not None:
                        bss.append(actual_bs)

                    # Print output
                    if print_every > 0 and (it % print_every) == 0:
                        t = datetime.datetime.now()
                        since = time.strftime(
                            "%H:%M:%S", time.gmtime((t - t0).total_seconds())
                        )
                        loss = sum(losses) / len(losses)
                        avg_bs = round(sum(bss) / len(bss))
                        losses = []
                        bss = []
                        print(
                            f"[Epoch {epoch_it}] it={it}, "
                            f"train loss={loss:.3f}, time={since}, "
                            f"actual/real batch={avg_bs}/{batch_size}"
                        )

                    # Visualize output
                    if visualize_every > 0 and (it % visualize_every) == 0:
                        print("Visualizing")
                        for data_vis in data_vis_list:
                            if cfg["generation"]["sliding_window"]:
                                out = generator.generate_mesh_sliding(data_vis["data"])
                            else:
                                out = generator.generate_mesh(data_vis["data"])
                            # Get statistics
                            try:
                                mesh, stats_dict = out
                            except TypeError:
                                mesh, stats_dict = out, {}

                            if isinstance(mesh, tuple):
                                mesh, _ = mesh

                            mesh.export(
                                os.path.join(
                                    out_dir,
                                    "vis",
                                    f"{it}_{data_vis['category']}_{data_vis['it']}.obj",
                                )
                            )

                    # Save checkpoint
                    if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                        print("Saving checkpoint")
                        checkpoint_io.save(
                            "model.pt",
                            epoch_it=epoch_it,
                            it=it,
                            loss_val_best=metric_val_best,
                        )

                    # Backup if necessary
                    if backup_every > 0 and (it % backup_every) == 0:
                        print("Backup checkpoint")
                        checkpoint_io.save(
                            f"model_{it}.pt",
                            epoch_it=epoch_it,
                            it=it,
                            loss_val_best=metric_val_best,
                        )
                    # Run validation
                    if validate_every > 0 and (it % validate_every) == 0:
                        eval_dict = trainer.evaluate(val_loader)
                        metric_val = eval_dict[model_selection_metric]
                        print(
                            f"Model selection ({model_selection_metric}): {metric_val}"
                        )

                        print("Validation results:", eval_dict)

                        for k, v in eval_dict.items():
                            logger.add_scalar(f"val/{k}", v, it)

                        if model_selection_sign * (metric_val - metric_val_best) > 0:
                            metric_val_best = metric_val
                            print(f"New best model (loss {metric_val_best})")
                            checkpoint_io.save(
                                "model_best.pt",
                                epoch_it=epoch_it,
                                it=it,
                                loss_val_best=metric_val_best,
                            )

                    # Exit if necessary
                    if exit_after > 0 and (datetime.datetime.now() - t0) >= exit_after:
                        print("Time limit reached. Exiting.")
                        checkpoint_io.save(
                            "model.pt",
                            epoch_it=epoch_it,
                            it=it,
                            loss_val_best=metric_val_best,
                        )
                        exit(3)
                if scheduler.get_last_lr()[0] > 5e-5:
                    scheduler.step()
            except Exception as e:
                print(f"ERROR (main) {e.__doc__}: {e}")
                pass
    else:
        print("Testing")
        eval_dict = trainer.evaluate(val_loader)
        metric_val = eval_dict[model_selection_metric]
        print("Test results:", eval_dict)
        print("Visualizing")
        for data_vis in data_vis_list:
            if cfg["generation"]["sliding_window"]:
                out = generator.generate_mesh_sliding(data_vis["data"])
            else:
                out = generator.generate_mesh(data_vis["data"])
            # Get statistics
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}

            mesh.export(
                os.path.join(
                    out_dir,
                    "vis",
                    f"test_vis_{data_vis['category']}_{data_vis['it']}.obj",
                )
            )
        exit()
