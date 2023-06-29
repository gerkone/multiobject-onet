import time

import numpy as np
import torch
import torch.optim as optim
import trimesh
from mcubes import marching_cubes
from torch import autograd
from tqdm import trange

from src.common import add_key, coord2index, make_3d_grid, normalize_coord

from src.utils.libmise import MISE
from src.utils.libsimplify import simplify_mesh

counter = 0

COLORS = [
    # red
    [255, 0, 0, 255],
    # green
    [0, 255, 0, 255],
    # blue
    [0, 0, 255, 255],
    # yellow
    [255, 255, 0, 255],
    # pink
    [255, 0, 255, 255],
    # light blue
    [0, 255, 255, 255],
    # light green
    [0, 255, 128, 255],
    # orange
    [255, 128, 0, 255],
    # purple
    [128, 0, 255, 255],
    # black
    [0, 0, 0, 255],
    # grey
    [128, 128, 128, 255],
]


class MultiObjectGenerator3D(object):
    """Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        multimesh (bool): whether to generate a mesh for each object
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    """

    def __init__(
        self,
        model,
        points_batch_size=100000,
        threshold=0.5,
        multimesh=False,
        refinement_step=0,
        device=None,
        resolution0=16,
        upsampling_steps=3,
        with_normals=False,
        padding=0.1,
        sample=False,
        input_type=None,
        vol_info=None,
        vol_bound=None,
        simplify_nfaces=None,
    ):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.multimesh = multimesh
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces

        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_mesh(self, data, return_stats=True, object_transforms=None):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get("inputs", torch.empty(1, 0)).to(device)
        # TODO get from segmentation
        node_tag = data.get("inputs.node_tags").to(device)  # (bs, pc)

        t0 = time.time()

        # obtain features for all crops
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            c = self.encode_crop(inputs, device)
        else:  # input the entire volume
            inputs = add_key(
                inputs, data.get("inputs.ind"), "points", "index", device=device
            )
            t0 = time.time()
            with torch.no_grad():
                c, obj_tag = self.model.encode_multi_object(inputs, node_tag)

        stats_dict["time (encode inputs)"] = time.time() - t0
        mesh = self.generate_from_latent(
            c, stats_dict=stats_dict, object_transforms=object_transforms, node_tag=obj_tag, obj_wise_occ=self.multimesh
        )

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, c=None, stats_dict={}, object_transforms=None, **kwargs):
        """Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
        threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        n_obj = kwargs.get("node_tag").max() + 1 if self.multimesh else 1

        # Shortcut
        if self.upsampling_steps == 0:
            raise NotImplementedError
            nx = self.resolution0
            pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            extractors = [
                MISE(self.resolution0, self.upsampling_steps, threshold)
                for _ in range(n_obj)
            ]
            obj_points = []
            # First pass
            start_points = extractors[0].query()
            # Query points
            pointsf = start_points / extractors[0].resolution
            # Normalize to bounding box
            pointsf = box_size * (pointsf - 0.5)
            pointsf = torch.FloatTensor(pointsf).to(self.device)
            # Evaluate model and update
            values = [
                v.squeeze(0).cpu().numpy()
                for v in self.eval_points(pointsf, c, **kwargs)
            ]
            assert len(values) == n_obj, "Object in scene mismatch."
            for i, (extr, v) in enumerate(zip(extractors, values)):
                v = v.astype(np.float64)
                extr.update(start_points, v)
                obj_points.append(extr.query())

            # TODO is there a smarter way without modifying MISE?
            # Mesh refinement
            for i, (extr, p) in enumerate(zip(extractors, obj_points)):
                while p.shape[0] != 0:
                    # Query points
                    pointsf = p / extr.resolution
                    # Normalize to bounding box
                    pointsf = box_size * (pointsf - 0.5)
                    pointsf = torch.FloatTensor(pointsf).to(self.device)
                    # Evaluate model objectwise and update
                    v = (
                        self.eval_points(pointsf, c, obj_i=i, **kwargs)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                    v = v.astype(np.float64)
                    extr.update(p, v)
                    p = extr.query()

            value_grids = [extr.to_dense() for extr in extractors]

        # Extract mesh
        stats_dict["time (eval points)"] = time.time() - t0

        object_meshes = self.extract_meshes(
            value_grids, c, stats_dict=stats_dict, color=self.multimesh
        )
        object_meshes = self.transform_objects(object_meshes, object_transforms)
        scene_mesh = trimesh.util.concatenate(object_meshes)
        return scene_mesh

    def eval_points(self, p, c=None, obj_i=None, **kwargs):
        """Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
        """
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode_multi_object(pi, c, **kwargs).logits
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        occ_hat = torch.cat(occ_hats, dim=-1)  # (pc,) or (obj, pc)
        # object-wise list of occupancy
        if self.multimesh:
            occ_hat = list(occ_hat.split(1, dim=0))
        else:
            occ_hat = [occ_hat]

        if obj_i is not None:
            occ_hat = occ_hat[obj_i]

        return occ_hat

    def generate_mesh_sliding(self, data, return_stats=True):
        raise NotImplementedError

    def get_crop_bound(self, inputs):
        """Divide a scene into crops, get boundary for each crop

        Args:
            inputs (dict): input point cloud
        """
        query_crop_size = self.vol_bound["query_crop_size"]
        input_crop_size = self.vol_bound["input_crop_size"]

        lb = inputs.min(axis=1).values[0].cpu().numpy() - 0.01
        ub = inputs.max(axis=1).values[0].cpu().numpy() + 0.01
        lb_query = (
            np.mgrid[
                lb[0] : ub[0] : query_crop_size,
                lb[1] : ub[1] : query_crop_size,
                lb[2] : ub[2] : query_crop_size,
            ]
            .reshape(3, -1)
            .T
        )
        ub_query = lb_query + query_crop_size
        center = (lb_query + ub_query) / 2
        lb_input = center - input_crop_size / 2
        ub_input = center + input_crop_size / 2
        # number of crops alongside x,y, z axis
        self.vol_bound["axis_n_crop"] = np.ceil((ub - lb) / query_crop_size).astype(int)
        # total number of crops
        num_crop = np.prod(self.vol_bound["axis_n_crop"])
        self.vol_bound["n_crop"] = num_crop
        self.vol_bound["input_vol"] = np.stack([lb_input, ub_input], axis=1)
        self.vol_bound["query_vol"] = np.stack([lb_query, ub_query], axis=1)

    def encode_crop(self, inputs, device, vol_bound=None):
        raise NotImplementedError

    def predict_crop_occ(self, pi, c, vol_bound=None, **kwargs):
        """Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        """
        occ_hat = pi.new_empty((pi.shape[0]))

        if pi.shape[0] == 0:
            return occ_hat
        pi_in = pi.unsqueeze(0)
        pi_in = {"p": pi_in}
        p_n = {}
        for key in self.vol_bound["fea_type"]:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = (
                normalize_coord(pi.clone(), vol_bound["input_vol"], plane=key)
                .unsqueeze(0)
                .to(self.device)
            )
        pi_in["p_n"] = p_n

        # predict occupancy of the current crop
        with torch.no_grad():
            occ_cur = self.model.decode(pi_in, c, **kwargs).logits
        occ_hat = occ_cur.squeeze(0)

        return occ_hat

    def extract_meshes(self, objwise_occ_hat, c=None, color=False, stats_dict=dict()):
        """Extracts the mesh from the predicted occupancy grid.

        Args:
            objwise_occ_hat (list): list of object-wise occupancy grids
            c (tensor): encoded feature volumes
            color (bool): whether to color the mesh. Different colors for different objects
            stats_dict (dict): stats dictionary
        """
        object_meshes = []
        for i, occ_hat in enumerate(objwise_occ_hat):
            # Some short hands
            n_x, n_y, n_z = occ_hat.shape
            box_size = 1 + self.padding
            threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)
            # Make sure that mesh is watertight
            t0 = time.time()
            occ_hat_padded = np.pad(occ_hat, 1, "constant", constant_values=-1e6)
            vertices, triangles = marching_cubes(occ_hat_padded, threshold)
            stats_dict["time (marching cubes)"] = time.time() - t0
            # Strange behaviour in libmcubes: vertices are shifted by 0.5
            # TODO investigate if its still the case
            # vertices -= 0.5
            # # Undo padding
            vertices -= 1

            if self.vol_bound is not None:
                # Scale the mesh back to its original metric
                bb_min = self.vol_bound["query_vol"][:, 0].min(axis=0)
                bb_max = self.vol_bound["query_vol"][:, 1].max(axis=0)
                mc_unit = max(bb_max - bb_min) / (
                    self.vol_bound["axis_n_crop"].max()
                    * self.resolution0
                    * 2**self.upsampling_steps
                )
                vertices = vertices * mc_unit + bb_min
            else:
                # Normalize to bounding box
                vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
                vertices = box_size * (vertices - 0.5)

            # Estimate normals if needed
            if self.with_normals and not vertices.shape[0] == 0:
                t0 = time.time()
                normals = self.estimate_normals(vertices, c)
                stats_dict["time (normals)"] = time.time() - t0

            else:
                normals = None

            # Create mesh
            mesh = trimesh.Trimesh(
                vertices, triangles, vertex_normals=normals, process=False
            )

            # Directly return if mesh is empty
            if vertices.shape[0] == 0:
                return mesh

            # TODO: normals are lost here
            if self.simplify_nfaces is not None:
                t0 = time.time()
                mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.0)
                stats_dict["time (simplify)"] = time.time() - t0

            # Refine mesh
            if self.refinement_step > 0:
                t0 = time.time()
                self.refine_mesh(mesh, occ_hat, c)
                stats_dict["time (refine)"] = time.time() - t0

            # Color mesh
            if color:
                mesh.visual.vertex_colors = np.tile(
                    COLORS[i % len(COLORS)], (len(mesh.vertices), 1)
                )

            object_meshes.append(mesh)

        return object_meshes

    def transform_objects(self, object_meshes, object_transforms):
        if object_transforms is None or all([t is None for t in object_transforms]):
            return object_meshes
        
        transformed_meshes = []
        assert len(object_meshes) == len(object_transforms)
        for mesh, transform in zip(object_meshes, object_transforms):
            if transform is None:
                transformed_meshes.append(mesh)
            else:
                transformed_meshes.append(mesh.apply_transform(transform))
        
        return transformed_meshes

    def estimate_normals(self, vertices, c=None):
        """Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        """
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode_multi_object(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        """Refines the predicted mesh.

        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        """

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert n_x == n_y == n_z
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode_multi_object(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True
            )[0]

            normal_target = normal_target / (
                normal_target.norm(dim=1, keepdim=True) + 1e-10
            )
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
