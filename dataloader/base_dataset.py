import torch
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
import random

class BaseDataset(Dataset):
    """
    Base class for dataset
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def event_formatting(events):

        xs = torch.from_numpy(events[0].astype(np.float32))
        ys = torch.from_numpy(events[1].astype(np.float32))
        ts = torch.from_numpy(events[2].astype(np.float32))
        ps = torch.from_numpy(events[3].astype(np.float32))
        ts = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-6)
        return torch.stack([xs, ys, ts, ps])

    @staticmethod
    def frame_formatting(frame):

        return torch.from_numpy(frame.astype(np.uint8)).float().unsqueeze(0) / 255

    @staticmethod
    def binary_search_h5_dset(dset, x_list, l=None, r=None, side='left'):
        gt_t0_list = []
        gt_index = 0
        num = len(x_list)
        for i in range(len(dset)):
            if dset[i] >= x_list[gt_index]:
                gt_t0_list.append(i)
                gt_index += 1

            if gt_index == num:
                break

        return gt_t0_list
