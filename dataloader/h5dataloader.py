from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
import torch
from tqdm import tqdm
import os
import pandas as pd
from copy import deepcopy
import yaml
# import MinkowskiEngine as ME
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

plt.style.use('seaborn-whitegrid')
# local modules
from dataloader.h5dataset import SequenceDataset, H5Dataset


def concatenate_datasets(data_file, dataset_type, dataset_config):
    """
    Generates a dataset for each data_path specified in data_file and concatenates the datasets.
    :param data_file: A file containing a list of paths to CTI h5 files.
                      Each file is expected to have a sequence of frame_{:09d}
    :param dataset_type: Pointer to dataset class
    :return ConcatDataset: concatenated dataset of all data_paths in data_file
    """
    data_paths = pd.read_csv(data_file, header=None).values.flatten().tolist()
    dataset_list = []
    print('Concatenating {} datasets'.format(dataset_type))
    for data_path in tqdm(data_paths):
        dataset_list.append(dataset_type(data_path, dataset_config))

    return ConcatDataset(dataset_list)


# ************************************************************************
class HDF5DataLoader(DataLoader):
    def __init__(self, dataloader_config):
        path_to_datalist_txt = dataloader_config['path_to_datalist_txt']
        self.dataset = concatenate_datasets(path_to_datalist_txt, H5Dataset, dataloader_config['dataset'])
        self.time_resolution = dataloader_config[
            'time_resolution'] if 'time_resolution' in dataloader_config.keys() else None
        self.gt_sensor_resolution = self.dataset.datasets[0].gt_sensor_resolution
        self.inp_sensor_resolution = self.dataset.datasets[0].inp_sensor_resolution
        self.scale = dataloader_config['dataset']['scale']

        if dataloader_config['use_ddp']:
            train_sampler = DistributedSampler(self.dataset, shuffle=dataloader_config['shuffle'])
            super().__init__(self.dataset,
                             batch_size=dataloader_config['batch_size'],
                             num_workers=dataloader_config['num_workers'],
                             pin_memory=dataloader_config['pin_memory'],
                             drop_last=dataloader_config['drop_last'],
                             sampler=train_sampler,
                             collate_fn=self.custom_collate,
                             worker_init_fn=self.worker_init_fn
                             )
        else:
            super().__init__(self.dataset,
                             batch_size=dataloader_config['batch_size'],
                             shuffle=dataloader_config['shuffle'],
                             num_workers=dataloader_config['num_workers'],
                             drop_last=dataloader_config['drop_last'],
                             pin_memory=dataloader_config['pin_memory'],
                             collate_fn=self.custom_collate,
                             worker_init_fn=self.worker_init_fn
                             )

    # @classmethod
    def custom_collate(self, batch):
        body = {}
        for key in batch[0].keys():
            body[key] = []
        for entry in batch:
            for key, value in entry.items():
                body[key].append(value)
        for key, value in deepcopy(body).items():
            if key in ['inp_events', 'inp_normalized_events', 'inp_scaled_events', 'inp_pol_mask', 'gt_events',
                       'gt_normalized_events']:
                body[key] = self.pack_tensor(value)
            else:
                body[key] = torch.stack(value)

        return body

    def normalize_events(self, events_list, resolution):
        """
        events_list: list, [N1x4, N2x4, ...]; torch.tensor, [x, y, t, p]
        """
        out_list = []
        for events in events_list:
            x, y, t, p = events[:, 0] / resolution[1], events[:, 1] / resolution[0], events[:, 2], events[:, 3]
            norm_events = torch.stack([x, y, t, p], dim=1).float()
            out_list.append(norm_events)

        out_tensor = self.pack_tensor(out_list)
        return out_tensor

    @staticmethod
    def pack_tensor(seq):
        """
        seq: [item0, item1, ...], item (torch.tensor): N x C
        return: B x N_max x C
        """
        output = []
        maxlen = 0
        for item in seq:
            maxlen = item.size(0) if item.size(0) > maxlen else maxlen
        for item in seq:
            tem = torch.zeros([maxlen, item.size(1)])
            tem[:item.size(0), :] = item
            output.append(tem)

        return torch.stack(output)

    @staticmethod
    def worker_init_fn(worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        pass


class InferenceHDF5DataLoader(DataLoader):
    def __init__(self, data_path, dataloader_config):
        self.dataset = H5Dataset(data_path, dataloader_config['dataset'])
        self.time_resolution = dataloader_config[
            'time_resolution'] if 'time_resolution' in dataloader_config.keys() else None
        self.gt_sensor_resolution = self.dataset.gt_sensor_resolution
        self.scale = dataloader_config['dataset']['scale']

        super().__init__(self.dataset,
                         batch_size=dataloader_config['batch_size'],
                         shuffle=dataloader_config['shuffle'],
                         num_workers=dataloader_config['num_workers'],
                         drop_last=dataloader_config['drop_last'],
                         pin_memory=dataloader_config['pin_memory'],
                         collate_fn=self.custom_collate,
                         worker_init_fn=self.worker_init_fn
                         )

    # @classmethod
    def custom_collate(self, batch):
        body = {}
        for key in batch[0].keys():
            body[key] = []
        for entry in batch:
            for key, value in entry.items():
                body[key].append(value)
        for key, value in deepcopy(body).items():
            if key in ['inp_events', 'inp_normalized_events', 'inp_scaled_events', 'inp_pol_mask', 'gt_events',
                       'gt_normalized_events']:
                body[key] = self.pack_tensor(value)
            else:
                body[key] = torch.stack(value)

        return body

    @staticmethod
    def pack_tensor(seq):
        """
        seq: [item0, item1, ...], item (torch.tensor): N x C
        return: B x N_max x C
        """
        output = []
        maxlen = 0
        for item in seq:
            maxlen = item.size(0) if item.size(0) > maxlen else maxlen
        for item in seq:
            tem = torch.zeros([maxlen, item.size(1)])
            tem[:item.size(0), :] = item
            output.append(tem)

        return torch.stack(output)

    @staticmethod
    def worker_init_fn(worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        pass


# ****************************************************************************
class HDF5DataLoaderSequence(DataLoader):
    def __init__(self, dataloader_config):
        path_to_datalist_txt = dataloader_config['path_to_datalist_txt']
        self.dataset = concatenate_datasets(path_to_datalist_txt, SequenceDataset, dataloader_config['dataset'])
        self.gt_sensor_resolution = self.dataset.datasets[0].gt_sensor_resolution
        self.inp_sensor_resolution = self.dataset.datasets[0].inp_sensor_resolution
        self.seqn = dataloader_config['dataset']['sequence']['seqn']

        if dataloader_config['use_ddp']:
            train_sampler = DistributedSampler(self.dataset, shuffle=dataloader_config['shuffle'])
            super().__init__(self.dataset,
                             batch_size=dataloader_config['batch_size'],
                             num_workers=dataloader_config['num_workers'],
                             pin_memory=dataloader_config['pin_memory'],
                             drop_last=dataloader_config['drop_last'],
                             sampler=train_sampler,
                             collate_fn=self.custom_collate,
                             worker_init_fn=self.worker_init_fn
                             )
        else:
            super().__init__(self.dataset,
                             batch_size=dataloader_config['batch_size'],
                             shuffle=dataloader_config['shuffle'],
                             num_workers=dataloader_config['num_workers'],
                             drop_last=dataloader_config['drop_last'],
                             pin_memory=dataloader_config['pin_memory'],
                             collate_fn=self.custom_collate,
                             worker_init_fn=self.worker_init_fn
                             )

    def custom_collate(self, batch):
        out = []
        sequence = []
        body = {}
        for key in batch[0][0].keys():
            body[key] = []
        for _ in range(len(batch[0])):
            sequence.append(deepcopy(body))
        for entry in batch:
            for i, item in enumerate(entry):
                for key in item.keys():
                    sequence[i][key].append(item[key])
        for entry in sequence:
            for key in entry.keys():
                if key in ['inp_events', 'inp_normalized_events', 'inp_scaled_events', 'inp_pol_mask', 'gt_events',
                           'gt_normalized_events']:
                    entry[key] = self.pack_tensor(entry[key])
                else:
                    entry[key] = torch.stack(entry[key])

        assert len(sequence) >= self.seqn
        for i in range(len(sequence) - self.seqn + 1):
            out.append(self.concat_dict(sequence[i:i + self.seqn]))

        return out

    @staticmethod
    def concat_dict(dicts):
        out = defaultdict(list)

        for key in dicts[0].keys():
            for item in dicts:
                out[key].append(item[key])

        for key in out.keys():
            out[key] = torch.stack(out[key], dim=1)

        return dict(out)

    @staticmethod
    def pack_tensor(seq):
        """
        seq: [item0, item1, ...], item (torch.tensor): N x C
        return: B x N_max x C
        """
        output = []
        maxlen = 0
        for item in seq:
            maxlen = item.size(0) if item.size(0) > maxlen else maxlen
        for item in seq:
            tem = torch.zeros([maxlen, item.size(1)])
            tem[:item.size(0), :] = item
            output.append(tem)

        return torch.stack(output)

    @staticmethod
    def worker_init_fn(worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        pass


class InferenceHDF5DataLoaderSequence(DataLoader):
    def __init__(self, data_path, dataloader_config):
        self.dataset = SequenceDataset(data_path, dataloader_config['dataset'])
        self.gt_sensor_resolution = self.dataset.gt_sensor_resolution
        self.inp_sensor_resolution = self.dataset.inp_sensor_resolution
        self.seqn = dataloader_config['dataset']['sequence']['seqn']
        self.scale = dataloader_config['dataset']['scale']

        super().__init__(self.dataset,
                         batch_size=dataloader_config['batch_size'],
                         shuffle=dataloader_config['shuffle'],
                         num_workers=dataloader_config['num_workers'],
                         drop_last=dataloader_config['drop_last'],
                         pin_memory=dataloader_config['pin_memory'],
                         collate_fn=self.custom_collate,
                         worker_init_fn=self.worker_init_fn
                         )

    def custom_collate(self, batch):
        out = []
        sequence = []
        body = {}
        for key in batch[0][0].keys():
            body[key] = []
        for _ in range(len(batch[0])):
            sequence.append(deepcopy(body))
        for entry in batch:
            for i, item in enumerate(entry):
                for key in item.keys():
                    sequence[i][key].append(item[key])
        for entry in sequence:
            for key in entry.keys():
                if key in ['inp_events', 'inp_normalized_events', 'inp_scaled_events', 'inp_pol_mask', 'gt_events',
                           'gt_normalized_events']:
                    entry[key] = self.pack_tensor(entry[key])
                else:
                    entry[key] = torch.stack(entry[key])

        assert len(sequence) >= self.seqn
        for i in range(len(sequence) - self.seqn + 1):
            out.append(self.concat_dict(sequence[i:i + self.seqn]))

        return out

    @staticmethod
    def concat_dict(dicts):
        out = defaultdict(list)

        for key in dicts[0].keys():
            for item in dicts:
                out[key].append(item[key])

        for key in out.keys():
            out[key] = torch.stack(out[key], dim=1)

        return dict(out)

    @staticmethod
    def pack_tensor(seq):
        """
        seq: [item0, item1, ...], item (torch.tensor): N x C
        return: B x N_max x C
        """
        output = []
        maxlen = 0
        for item in seq:
            maxlen = item.size(0) if item.size(0) > maxlen else maxlen
        for item in seq:
            tem = torch.zeros([maxlen, item.size(1)])
            tem[:item.size(0), :] = item
            output.append(tem)

        return torch.stack(output)

    @staticmethod
    def worker_init_fn(worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        pass




