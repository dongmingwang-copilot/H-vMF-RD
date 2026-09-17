import torch
from torch.utils.data import Sampler

class FixedStepBatches(Sampler):
    """Epoch permutations depend only on the seed, including after resume."""

    def __init__(self, count, batch, seed, start, stop):
        self.count, self.batch, self.seed = count, batch, seed
        self.start, self.stop = start, stop
        self.per_epoch = count // batch
        assert self.per_epoch > 0

    def __iter__(self):
        previous = None
        for step in range(self.start, self.stop):
            epoch, position = divmod(step, self.per_epoch)
            if epoch != previous:
                permutation = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                previous = epoch
            yield permutation[position * self.batch:(position + 1) * self.batch]

    def __len__(self):
        return self.stop - self.start
