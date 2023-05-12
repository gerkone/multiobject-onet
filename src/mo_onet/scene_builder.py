class SceneBuilder:
    """Build the scene from a list of meshes.

    Denormalize and shift local frames to scene frame.
    """

    def __init__(self, metadata):
        self.metadata = metadata

    def __call__(self, meshes):
        # TODO
        raise NotImplementedError
