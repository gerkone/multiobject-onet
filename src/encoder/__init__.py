from src.encoder import egnn, pointnet, pointnetpp, voxels, gconv

encoder_dict = {
    "pointnet_local_pool": pointnet.LocalPoolPointnet,
    "pointnet_crop_local_pool": pointnet.PatchLocalPoolPointnet,
    "pointnet_plus_plus": pointnetpp.PointNetPlusPlus,
    "voxel_simple_local": voxels.LocalVoxelEncoder,
    "egnn": egnn.EGNN,
    "gconv": gconv.MOGConv,
    "e3_gconv": gconv.MOE3GConv,
}
