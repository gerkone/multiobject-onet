from torch import nn

# TODO instance segmentation


class Segmenter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    @staticmethod
    def filter_state_dict(ckp, keep_first=False, keep_last=False):
        pass
