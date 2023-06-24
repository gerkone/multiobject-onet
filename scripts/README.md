# Synthetic Indoor Scene Dataset
For scene-level reconstruction, we create a synthetic dataset of 5000
scenes with multiple objects from ShapeNet (chair, sofa, lamp, cabinet, table). There are also ground planes.

## Download the original dataset
You can download our preprocessed data (144 GB) using

```
bash scripts/download_data.sh
```

This script should download and unpack the data automatically into the `data/synthetic_room_dataset` folder.  

## Process synthetic_room_dataset
To extract watertight meshes with `scripts/binvox_to_off.py`.

## Generate synthetic rooms
You can now generate the rooms with both semantic and instance labels with `scripts/build_synthetic_scene.py`. Update with your `input_path_watertight`.