# python
# File: `data_processing.py`
import torch
from typing import Iterator, List

Tensor = torch.Tensor

class DataLoader:
    """
    Minimal DataLoader:
    - Initialize with x and a required batch_size.
    - Iteration yields consecutive slices of x preserving device/dtype.
    """
    def __init__(self, x: Tensor, batch_size: int):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an int")
        self.x = x
        self.N = int(x.shape[0])
        if self.N == 0:
            # empty dataset -> no batches
            self.batch_size = 0
        else:
            if batch_size <= 0:
                raise ValueError("batch_size must be > 0")
            self.batch_size = int(batch_size)

    def __iter__(self) -> Iterator[Tensor]:
        if self.N == 0 or self.batch_size == 0:
            return iter(())
        return (self.x[start:min(start + self.batch_size, self.N)]
                for start in range(0, self.N, self.batch_size))

    def to_list(self) -> List[Tensor]:
        return list(self)

def slice_into_batches(x: Tensor, batch_size: int) -> List[Tensor]:
    """
    Convenience function: slice x into batches using `batch_size`.
    """
    return DataLoader(x, batch_size=batch_size).to_list()

