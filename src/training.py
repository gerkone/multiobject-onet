from collections import defaultdict

import time
import numpy as np


class BaseTrainer(object):
    """Base trainer class."""

    def evaluate(self, val_loader):
        """Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        """
        eval_list = defaultdict(list)
        tt = 0.0
        for data in val_loader:
            st = time.perf_counter_ns()
            eval_step_dict = self.eval_step(data)
            tt += (time.perf_counter_ns() - st) / 1e6

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        eval_dict["time"] = tt / len(val_loader)
        return eval_dict

    def train_step(self, *args, **kwargs):
        """Performs a training step."""
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """Performs an evaluation step."""
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """Performs  visualization."""
        raise NotImplementedError
