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


class MultiObjectGenerator3D(object):
    """Generator class for the multi-object case.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        scene_builder (nn.Module): scene builder
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
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
        scene_builder,
        points_batch_size=100000,
        threshold=0.5,
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
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.scene_builder = scene_builder

        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_mesh(self, data, return_stats=True):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get("inputs", torch.empty(1, 0)).to(device)
        kwargs = {}

        t0 = time.time()

        # obtain features for all crops
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            code = self.encode_crop(inputs, device)
        else:  # input the entire volume
            inputs = add_key(
                inputs, data.get("inputs.ind"), "points", "index", device=device
            )
            t0 = time.time()
            with torch.no_grad():
                codes = self.model.encode_and_segment(inputs)
        stats_dict["time (encode inputs)"] = time.time() - t0

        # generate list of meshes
        meshes = self.generate_from_latent(codes, stats_dict=stats_dict, **kwargs)

        # scene building. from local to global
        scene = self.scene_builder(meshes)

        if return_stats:
            return scene, stats_dict
        else:
            return scene

    def generate_mesh_sliding(self, data, return_stats=True):
        raise NotImplementedError

    def generate_from_latent(self, codes=None, stats_dict={}, **kwargs):
        """Generates full scene mesches from list of latent codes."""
        return [
            self._generate_sigle_from_latent(c, stats_dict, **kwargs) for c in codes
        ]

    def _generate_sigle_from_latent(self, c=None, stats_dict={}, **kwargs):
        """Generates one mesh from latent. Works for shapes normalized to a unit cube.

        Args:
            c (tensor): latent conditioned c for a single object
            stats_dict (dict): stats dictionary
        """
        threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            # grid points in normalized coordinates
            pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)

            values = self._eval_points_single_shape(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # Evaluate model and update
                values = (
                    self._eval_points_single_shape(pointsf, c, **kwargs).cpu().numpy()
                )
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict["time (eval points)"] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

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
        """Encode a crop to feature volumes

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        """
        if vol_bound is None:
            vol_bound = self.vol_bound

        index = {}
        for fea in self.vol_bound["fea_type"]:
            # crop the input point cloud
            mask_x = (inputs[:, :, 0] >= vol_bound["input_vol"][0][0]) & (
                inputs[:, :, 0] < vol_bound["input_vol"][1][0]
            )
            mask_y = (inputs[:, :, 1] >= vol_bound["input_vol"][0][1]) & (
                inputs[:, :, 1] < vol_bound["input_vol"][1][1]
            )
            mask_z = (inputs[:, :, 2] >= vol_bound["input_vol"][0][2]) & (
                inputs[:, :, 2] < vol_bound["input_vol"][1][2]
            )
            mask = mask_x & mask_y & mask_z

            p_input = inputs[mask]
            if p_input.shape[0] == 0:  # no points in the current crop
                p_input = inputs.squeeze()
                ind = coord2index(
                    p_input.clone(),
                    vol_bound["input_vol"],
                    reso=self.vol_bound["reso"],
                    plane=fea,
                )
                if fea == "grid":
                    ind[~mask] = self.vol_bound["reso"] ** 3
                else:
                    ind[~mask] = self.vol_bound["reso"] ** 2
            else:
                ind = coord2index(
                    p_input.clone(),
                    vol_bound["input_vol"],
                    reso=self.vol_bound["reso"],
                    plane=fea,
                )
            index[fea] = ind.unsqueeze(0)
            input_cur = add_key(
                p_input.unsqueeze(0), index, "points", "index", device=device
            )

        with torch.no_grad():
            codes = self.model.encode_and_segment(input_cur)
        return codes

    def _predict_crop_occ_single_shape(self, pi, c, vol_bound=None, **kwargs):
        """Predict occupancy values for a crop of a single object.

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
            occ_cur = self.model.decode_single(pi_in, c, **kwargs).logits
        occ_hat = occ_cur.squeeze(0)

        return occ_hat

    def _eval_points_single_shape(self, p, c=None, vol_bound=None, **kwargs):
        """Evaluates the single shape occupancy values for the points.

        Args:
            p (tensor): points
            codes (tensor): encoded feature volumes
        """
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            if self.input_type == "pointcloud_crop":
                if self.vol_bound is not None:  # sliding-window manner
                    occ_hat = self._predict_crop_occ_single_shape(
                        pi, c, vol_bound=vol_bound, **kwargs
                    )
                    occ_hats.append(occ_hat)
                else:  # entire scene
                    pi_in = pi.unsqueeze(0).to(self.device)
                    pi_in = {"p": pi_in}
                    p_n = {}
                    for key in c.keys():
                        # normalized to the range of [0, 1]
                        p_n[key] = (
                            normalize_coord(pi.clone(), self.input_vol, plane=key)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                    pi_in["p_n"] = p_n
                    with torch.no_grad():
                        occ_hat = self.model.decode_single_object(
                            pi_in, c, **kwargs
                        ).logits
                    occ_hats.append(occ_hat.squeeze(0).detach().cpu())
            else:
                pi = pi.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    occ_hat = self.model.decode_single_object(pi, c, **kwargs).logits
                occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        """Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        """
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
        # TODO investigate
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

        return mesh

    def _estimate_normals_SINGLE_SHAPE(self, vertices, c=None):
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
            occ_hat = self.model.decode_single_object(vi, c).logits
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
            c (tensor): latent conditioned code
        """
        # TODO
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

        for _ in trange(self.refinement_step):
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
                self.model.decode_single_object(face_point.unsqueeze(0), c).logits
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