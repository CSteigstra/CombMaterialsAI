import torch
import numpy as np

class RollTransform:
    def __init__(self, cell_h=2, cell_w=2) -> None:
        self.cell_h = cell_h
        self.cell_w = cell_w
    
    def __call__(self, x, y):
        """Roll the input grid x by a random number of cells."""
        h, w = x.shape
        rh, rw = torch.randint(0, h, (1, )), torch.randint(0, w, (1, ))
        x = torch.roll(x, (rh * self.cell_h, rw * self.cell_w), dims=(0, 1))
        return x, y   

class ScaleTransform:
    """Repeat the input image by a random scale between n_min and n_max."""

    def __init__(self, n_min=1, n_max=6):
        self.n_min = n_min
        self.n_max = n_max

    def __call__(self, x, y):
        """Repeat the input image by a random scale between n_min and n_max."""
        h, w = x.shape
        r = torch.randint(self.n_min, self.n_max+1)
        x = x.repeat(r, r)

        # Pad to the maximum size.
        x = torch.nn.functional.pad(x, (0, w * self.n_max - w, 0, h * self.n_max - h))

        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[h * r:, w * r:] = 1

        return x, y * r, mask
        # Shuffle scales.
        r_scales = torch.randperm(self.n_scales) + 1
        
        # Split batch into n sub-batches, n=#scales.
        x = torch.tensor_split(x, self.n_scales, dim=0)
        
        # Repeat each sub-batch by the shuffled scales.
        return {f'scale_{s}': (_x.repeat(s, s), _y*s) for _x, _y, s in zip(x, y, r_scales)}
        