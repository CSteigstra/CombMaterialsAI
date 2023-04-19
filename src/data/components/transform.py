import torch
import numpy as np

class ScaleTransform:
    """Rotate by one of the given angles."""

    def __init__(self, n_scales=6):
        self.scales = torch.arange(1, n_scales + 1)

    def __call__(self, x, y):
        """Returns a dictionary containing sub-batches of x on each scale."""
        batch_size = x.shape[0]
        # TODO: Shuffle and scale y accordingly.
        # Shuffle scales.
        r_scales = self.scales[torch.randperm(len(self.scales))]

        # Split batch into n sub-batches, n=#scales.
        di = batch_size // len(self.scales)
        x = torch.split(x, di, dim=0)
        
        # Repeat each sub-batch by the shuffled scales.
        return {f'scale_{s}': _x.repeat(s, s) for _x, s in zip(x, r_scales)}
        