import torch
import numpy as np
from unicore.data import BaseWrapperDataset
from functools import lru_cache

def collate_tokens_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size - len(v):] if left_pad else res[i][:len(v), :len(v)])
    return res

def collate_tokens_3d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 3d tensor, 
    3d tensor is assume to have shape (n,n,3), 
    last dimension is not padded."""
    size = max(v.size(0) for v in values)
    d = values[0].size(-1)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size, d).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size - len(v):, :] if left_pad else res[i][:len(v), :len(v), :])
    return res

def collate_cross_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_length is not None:
        raise NotImplementedError("pad_to_length is not implemented for cross padding.")
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size_h - v.size(0) :, size_w - v.size(1) :]
            if left_pad
            else res[i][: v.size(0), : v.size(1)],
        )
    return res

def collate_cross_3d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 3d tensors into a padded 3d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    d = values[0].size(-1)
    if pad_to_length is not None:
        raise NotImplementedError("pad_to_length is not implemented for cross padding.")
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w, d).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size_h - v.size(0) :, size_w - v.size(1) :, :]
            if left_pad
            else res[i][: v.size(0), : v.size(1), :],
        )
    return res

class DisplacementDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
    
    @lru_cache(maxsize = 16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3).numpy()
        displacement = pos[:, None, :] - pos[None, :, :]
        return torch.from_numpy(displacement.astype(np.float32))

class CrossDisplacementDataset(BaseWrapperDataset):
    def __init__(self, mol_dataset, pocket_dataset):
        super().__init__(mol_dataset)
        self.mol_dataset = mol_dataset
        self.pocket_dataset = pocket_dataset
    
    @lru_cache(maxsize = 16)
    def __getitem__(self, idx):
        mol_pos = self.mol_dataset[idx].view(-1, 3).numpy()
        pocket_pos = self.pocket_dataset[idx].view(-1, 3).numpy()
        cross_displacement = mol_pos[:, None, :] - pocket_pos[None, :, :]
        return torch.from_numpy(cross_displacement.astype(np.float32))

class FullDisplacementDataset(BaseWrapperDataset):
    def __init__(self, mol_dataset, pocket_dataset):
        super().__init__(mol_dataset)
        self.mol_dataset = mol_dataset
        self.pocket_dataset = pocket_dataset
    
    @lru_cache(maxsize = 16)
    def __getitem__(self, idx):
        mol_pos = self.mol_dataset[idx].view(-1, 3).numpy()
        pocket_pos = self.pocket_dataset[idx].view(-1, 3).numpy()
        all_pos = np.concatenate((mol_pos, pocket_pos), axis = 0)
        #get r_ij = x_i - x_j
        displacement = all_pos[:, None, :] - all_pos[None, :, :]
        return torch.from_numpy(displacement.astype(np.float32))

class RightPadDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_2d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class RightPadDataset3D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_3d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class PrependAndAppend3DDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            h,w,d = item.size(-3), item.size(-2), item.size(-1) #d is the dimension of the coordinate and is not padded
            new_item = torch.full((h + 2, w + 2, d), self.token).type_as(item)
            new_item[1:-1, 1:-1, :] = item 
            return new_item
        return item

class RightPadDatasetCross2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_cross_2d(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8
        )

class RightPadDatasetCross3D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_cross_3d(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8
        )