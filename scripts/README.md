# Synthetic Indoor Scene Dataset
For scene-level reconstruction, we create a synthetic dataset with multiple objects from ShapeNet (chair, sofa, lamp, cabinet, table). There are also ground planes.

## Download the original dataset
You can download the preprocessed ShapeNet data (80GB) using

```
bash scripts/download_data.sh
```

This script should download and unpack the data automatically into `data/synthetic_room_dataset`.  

## Process synthetic_room_dataset
To extract watertight meshes with 
```
python scripts/binvox_to_off.py --inputs=<ShapeNet directory path> --outputs=<watertight directory path>
```

## Generate synthetic rooms
You can now generate the rooms with both semantic and instance labels with 
```
python scripts/build_synthetic_scene.py --input=<ShapeNet directory path> --input-watertight=<watertight directory path> --output=<synthetic rooms output path>
```