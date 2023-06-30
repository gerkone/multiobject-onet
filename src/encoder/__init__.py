from src.encoder import egnn, pointnet, pointnetpp, voxels, gconv

encoder_dict = {
    # ONet
    "pointnet_resnet": pointnet.ResnetPointnet,
    # ConvONet
    "pointnet_local_pool": pointnet.LocalPoolPointnet,
    "pointnet_crop_local_pool": pointnet.PatchLocalPoolPointnet,
    "pointnet_plus_plus": pointnetpp.PointNetPlusPlus,
    "voxel_simple_local": voxels.LocalVoxelEncoder,
    # MOONet
    "egnn": egnn.EGNN,
    "gconv": gconv.MOGConv,
}
