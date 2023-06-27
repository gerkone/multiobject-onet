import argparse
import trimesh
import sys
import os
import glob
import tqdm

sys.path.append(".")
from src.utils import binvox_rw

parser = argparse.ArgumentParser("Convert .binvox files to .off")
parser.add_argument("--inputs", "-i", type=str, help="Path to input dir with voxels.")
parser.add_argument(
    "--outputs", "-o", type=str, help="Path to output dir with watertight meshes."
)
args = parser.parse_args()

input_dir = args.inputs
output_dir = args.outputs

input_paths = glob.glob(os.path.join(input_dir, "**", "*.binvox"), recursive=True)

for input_path in tqdm.tqdm(input_paths):
    with open(input_path, "rb") as f:
        binvox_data = binvox_rw.read_as_3d_array(f)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        binvox_data.data, pitch=binvox_data.scale
    )
    output_path = os.path.join(
        output_dir, os.path.relpath(input_path, input_dir).replace(".binvox", ".off")
    )
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    mesh.export(output_path)
