import torch
import nqs_playground._C_nqs as _C


class SpinDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping spin configurations and corresponding values.

    Each sample will be retrieved by indexing along the first dimension.

    .. note:: This class is very similar to :py:class:`torch.utils.data.TensorDataset`
              except that ``spins`` is a NumPy array of structured type and is
              thus incompatible with :py:class:`torch.utils.data.TensorDataset`.

    :param spins: a NumPy array of :py:class:`CompactSpin`.
    :param values: a NumPy array or a Torch tensor.
    :param bool unpack: if ``True``, ``spins`` will be unpacked into Torch tensors. 
    """

    def __init__(self, spins, values, unpack=False):
        size = lambda x, i: x.size(i) if isinstance(x, torch.Tensor) else x.shape[i]
        if size(spins, 0) != size(values, 0):
            raise ValueError(
                "spins and values must have the same size of the first dimension, but {} != {}".format(
                    size(spins, 0), size(values, 0)
                )
            )
        if not isinstance(unpack, bool):
            raise ValueError(
                "unpack should be a boolean, but got unpack={}".format(unpack)
            )
        self.spins = spins
        self.values = values
        self.unpack = unpack
        self._size = size(spins, 0)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index):
        if self.unpack:
            if isinstance(index, int):
                return _C.unsafe_get(self.spins, index), self.values[index]
            return _C.unpack(self.spins[index]), self.values[index]
        return self.spins[index], self.values[index]


class BatchedRandomSampler(torch.utils.data.Sampler):
    r"""Samples batches of elements randomly.

    :param int num_samples: number of elements in the dataset
    :param int batch_size: size of mini-batch
    :param bool drop_last: if ``True``, the sampler will ignore the last batch
        if its size would be less than ``batch_size``.

    Example:
        >>> list(BatchedRandomSampler(10, batch_size=3, drop_last=True))
        [tensor([6, 8, 0]), tensor([2, 3, 1]), tensor([4, 7, 9])]
        >>> list(BatchedRandomSampler(10, batch_size=3, drop_last=False))
        [tensor([3, 4, 7]), tensor([0, 2, 8]), tensor([1, 5, 9]), tensor([6])]
    """

    def __init__(self, num_samples: int, batch_size: int, drop_last: bool):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._indices = None

        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer, but got num_samples={}".format(
                    num_samples
                )
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer, but got batch_size={}".format(
                    batch_size
                )
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean, but got drop_last={}".format(drop_last)
            )

    def __iter__(self):
        if self._indices is None:
            self._indices = torch.randperm(self.num_samples)
        else:
            torch.randperm(self.num_samples, out=self._indices)
        i = 0
        while i + self.batch_size <= self.num_samples:
            yield self._indices[i : i + self.batch_size]
            i += self.batch_size
        if i < self.num_samples and not self.drop_last:
            yield self._indices[i:]

    def __len__(self) -> int:
        """
        Returns the number of batches that will be produced by :py:func:`__iter__`.
        """
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def make_spin_dataloader(
    spins, values, batch_size: int, drop_last: bool = False, unpack: bool = True
):
    r"""Creates a new dataloader from packed spin configurations and corresponding
    values.
    """
    dataset = SpinDataset(spins, values, unpack)
    sampler = BatchedRandomSampler(len(dataset), batch_size, drop_last)
    # This is a bit of a hack. We want to use our custom batch sampler, but don't
    # want PyTorch to use auto_collate. So we tell PyTorch that our sampler is
    # a plain sampler (i.e. not batch_sampler), but set batch_size to 1 to
    # disable automatic batching
    return torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=sampler, collate_fn=lambda x: x[0]
    )
