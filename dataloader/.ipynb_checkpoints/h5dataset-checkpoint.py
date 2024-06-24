import torch
from torch.utils.data import Dataset
import torch.nn.functional as func
import os
from glob import glob
import h5py
import cv2
from tqdm import tqdm
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

plt.style.use('seaborn-whitegrid')
# local modules
from dataloader.base_dataset import BaseDataset
from dataloader.encodings import *
from myutils.vis_events.visualization import *


class H5Dataset(BaseDataset):
    def __init__(self, h5_file_path, config):
        super().__init__()

        self.config = config
        self.h5_file_path = h5_file_path
        self.set_data_scale()
        self.load_metadata()
        self.set_data_mode()

    def set_data_scale(self):
        self.h5_file = h5py.File(self.h5_file_path, 'r')
        self.need_gt_events = self.config.get('need_gt_events', False)
        self.real_world_test = self.config.get('real_world_test', False)
        self.custom_resolution = self.config.get('custom_resolution', None)
        self.dataset_length = self.config.get('dataset_length', None)
        self.add_noise = self.config.get('add_noise', {'enabled': False})
        self.sensor_resolution = self.h5_file.attrs['sensor_resolution'].tolist()
        self.scale = self.config['scale']
        self.ori_scale = self.config['ori_scale']

        self.gt_sensor_resolution = None
        self.gt_prex = None
        if self.real_world_test:
            if self.ori_scale == 'down8' and not self.need_gt_events:
                self.inp_sensor_resolution = [round(i / 8) for i in self.sensor_resolution]
                self.inp_down_sensor_resolution = [round(i / self.scale) for i in self.inp_sensor_resolution]
                self.inp_prex = 'down8_real'
                self.gt_prex = self.inp_prex
                if self.scale == 2:
                    self.gt_sensor_resolution = [round(i / 4) for i in self.sensor_resolution]
                elif self.scale == 4:
                    self.gt_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
                elif self.scale == 8:
                    self.gt_sensor_resolution = self.sensor_resolution
                else:
                    self.gt_sensor_resolution = self.sensor_resolution
            else:
                raise Exception(f'Error real world test!')

        elif self.ori_scale == 'ori':
            self.inp_sensor_resolution = self.sensor_resolution
            self.inp_down_sensor_resolution = [round(i / self.scale) for i in self.inp_sensor_resolution]
            self.inp_prex = 'ori'
            if not self.need_gt_events:
                self.gt_sensor_resolution = [round(i * self.scale) for i in self.inp_sensor_resolution]
                self.gt_prex = self.inp_prex
            elif self.scale == 1:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex = 'ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down2':
            self.inp_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
            self.inp_down_sensor_resolution = [round(i / self.scale) for i in self.inp_sensor_resolution]
            self.inp_prex = 'down2'
            if not self.need_gt_events:
                self.gt_sensor_resolution = [round(i * self.scale) for i in self.inp_sensor_resolution]
                self.gt_prex = self.inp_prex
            elif self.scale == 2:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex = 'ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down4':
            self.inp_sensor_resolution = [round(i / 4) for i in self.sensor_resolution]
            self.inp_down_sensor_resolution = [round(i / self.scale) for i in self.inp_sensor_resolution]
            self.inp_prex = 'down4'
            if not self.need_gt_events:
                self.gt_sensor_resolution = [round(i * self.scale) for i in self.inp_sensor_resolution]
                self.gt_prex = self.inp_prex
            elif self.scale == 2:
                self.gt_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
                self.gt_prex = 'down2'
            elif self.scale == 4:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex = 'ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down8':
            self.inp_sensor_resolution = [round(i / 8) for i in self.sensor_resolution]
            self.inp_down_sensor_resolution = [round(i / self.scale) for i in self.inp_sensor_resolution]
            self.inp_prex = 'down8'
            if not self.need_gt_events:
                self.gt_sensor_resolution = [round(i * self.scale) for i in self.inp_sensor_resolution]
                self.gt_prex = self.inp_prex
            elif self.scale == 2:
                self.gt_sensor_resolution = [round(i / 4) for i in self.sensor_resolution]
                self.gt_prex = 'down4'
            elif self.scale == 4:
                self.gt_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
                self.gt_prex = 'down2'
            elif self.scale == 8:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex = 'ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        elif self.ori_scale == 'down16':
            self.inp_sensor_resolution = [round(i / 16) for i in self.sensor_resolution]
            self.inp_down_sensor_resolution = [round(i / self.scale) for i in self.inp_sensor_resolution]
            self.inp_prex = 'down16'
            if not self.need_gt_events:
                self.gt_sensor_resolution = [round(i * self.scale) for i in self.inp_sensor_resolution]
                self.gt_prex = self.inp_prex
            elif self.scale == 2:
                self.gt_sensor_resolution = [round(i / 8) for i in self.sensor_resolution]
                self.gt_prex = 'down8'
            elif self.scale == 4:
                self.gt_sensor_resolution = [round(i / 4) for i in self.sensor_resolution]
                self.gt_prex = 'down4'
            elif self.scale == 8:
                self.gt_sensor_resolution = [round(i / 2) for i in self.sensor_resolution]
                self.gt_prex = 'down2'
            elif self.scale == 16:
                self.gt_sensor_resolution = self.sensor_resolution
                self.gt_prex = 'ori'
            else:
                raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

        else:
            raise Exception(f'Error scale setting: scale {self.scale}, ori_scale {self.ori_scale}')

    def load_metadata(self):
        self.time_bins = self.config['time_bins']
        self.num_events = len(self.h5_file[f'{self.inp_prex}_events']['ts'][:])
        self.num_gt_events = len(self.h5_file[f'{self.gt_prex}_events']['ts'][:]) if self.need_gt_events else None
        self.t0 = self.h5_file[f'{self.inp_prex}_events']['ts'][0]
        self.tk = self.h5_file[f'{self.inp_prex}_events']['ts'][-1]
        self.duration = self.tk - self.t0
        self.hot_events = torch.zeros(self.inp_sensor_resolution)
        self.hot_idx = 0

        self.need_gt_frame = self.config.get('need_gt_frame', False)
        if self.need_gt_frame:
            self.gt_frame_ts = []
            for img_name in self.h5_file['ori_images']:
                self.gt_frame_ts.append(self.h5_file['ori_images/{}'.format(img_name)].attrs['timestamp'])

    def set_data_mode(self):
        self.data_mode = self.config['mode']
        self.window = self.config['window']
        self.sliding_window = self.config['sliding_window']

        if self.data_mode == 'events':
            max_length = max(int(self.num_events / (self.window - self.sliding_window)), 0)
            if self.dataset_length is not None:
                self.length = self.dataset_length if self.dataset_length <= max_length else max_length
            else:
                self.length = max_length
            self.event_indices, self.gt_event_indices = self.compute_k_indices()
        elif self.data_mode == 'time':
            max_length = max(int(self.duration / (self.window - self.sliding_window)), 0)
            if self.dataset_length is not None:
                self.length = self.dataset_length if self.dataset_length <= max_length else max_length
            else:
                self.length = max_length
            self.event_indices, self.gt_event_indices = self.compute_timeblock_indices()
        elif self.data_mode == 'frame':
            max_length = len(self.h5_file['ori_images']) - 1
            self.num_frames = len(self.h5_file['ori_images'])
            if self.dataset_length is not None:
                self.length = self.dataset_length if self.dataset_length <= max_length else max_length
            else:
                self.length = max_length
            self.event_indices, self.gt_event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid data mode chosen ({})".format(self.data_mode))

        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        gt_k_indices = []

        for i in range(self.__len__()):
            idx0 = (self.window - self.sliding_window) * i
            idx1 = idx0 + self.window
            if idx1 > self.num_events - 1:
                idx1 = self.num_events - 1
            k_indices.append([idx0, idx1])

        if self.need_gt_events:
            gt_k_indices = self.get_gt_event_indices_num(k_indices)

        return k_indices, gt_k_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        gt_timeblock_indices = []
        start_idx = 0

        for i in range(self.__len__()):
            start_time = ((self.window - self.sliding_window) * i) + self.t0
            end_time = start_time + self.window
            end_idx = self.find_ts_index(end_time)
            if self.need_gt_events:
                gt_idx0, gt_idx1 = self.get_gt_event_indices_num(start_idx, end_idx)
                gt_timeblock_indices.append([gt_idx0, gt_idx1])
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx

        return timeblock_indices, gt_timeblock_indices

    def compute_frame_indices(self):
        frame_indices = []
        gt_frame_indices = []
        start_idx = 0
        for ts in self.gt_frame_ts[:self.length]:
            end_idx = self.find_ts_index(ts)
            if self.need_gt_events:
                gt_idx0, gt_idx1 = self.get_gt_event_indices_num(start_idx, end_idx)
                # gt_idx0, gt_idx1 = self.get_gt_event_indices_time(start_idx, end_idx)
                gt_frame_indices.append([gt_idx0, gt_idx1])
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx

        return frame_indices, gt_frame_indices


    def find_ts_index(self, timestamp):
        idx = self.binary_search_h5_dset(self.h5_file[f'{self.inp_prex}_events/ts'][:], timestamp)
        if idx > self.num_events - 1:
            idx = self.num_events - 1

        return idx

    def __getitem__(self, index, Pause=False, seed=None):
        if seed is None:
            seed = random.randint(0, 2 ** 32)
        idx0, idx1 = self.get_event_indices(index)
        if self.need_gt_events:
            gt_idx0, gt_idx1 = self.get_gt_event_indices(index)

        # events
        inp_events = self.get_events(idx0, idx1)
        if self.config['data_augment']['enabled']:
            inp_events = self.augment_event(inp_events, self.inp_sensor_resolution, seed)
        inp_events_torch = self.event_formatting(inp_events)
        if self.need_gt_events:
            gt_events = self.get_gt_events(gt_idx0, gt_idx1)
            if self.config['data_augment']['enabled']:
                gt_events = self.augment_event(gt_events, self.gt_sensor_resolution, seed)
            gt_events_torch = self.event_formatting(gt_events)
        else:
            gt_events_torch = torch.zeros([4, 1])

        # add noise
        if self.add_noise['enabled']:
            noise = self.add_noise_event(self.window, self.inp_sensor_resolution, seed,
                                         noise_level=self.add_noise['noise_level'])
            inp_events_torch = torch.cat([inp_events_torch, noise], dim=1)

        # gt frame
        if self.need_gt_frame:
            gt_img = self.get_gt_frame(idx0, idx1)
            if self.config['data_augment']['enabled']:
                gt_img = self.augment_frame(gt_img, seed)
            gt_img_torch = self.frame_formatting(
                cv2.resize(gt_img, dsize=self.gt_sensor_resolution[::-1], interpolation=cv2.INTER_CUBIC))
            gt_img_inp_size_torch = self.frame_formatting(
                cv2.resize(gt_img, dsize=self.inp_sensor_resolution[::-1], interpolation=cv2.INTER_CUBIC))

        if self.data_mode == 'frame':
            frame = self.get_frame(index)
            if self.config['data_augment']['enabled']:
                frame = self.augment_frame(frame, seed)
            frame_torch = self.frame_formatting(
                cv2.resize(frame, dsize=self.gt_sensor_resolution[::-1], interpolation=cv2.INTER_CUBIC))

        # Pause
        if Pause:
            inp_events_torch = torch.zeros([4, 1])

        inp_event_cnt = self.create_cnt_encoding(inp_events_torch, self.inp_sensor_resolution)  # + inp_cnt_noise
        gt_event_cnt = self.create_cnt_encoding(gt_events_torch, self.gt_sensor_resolution)

        item = {
            'inp_cnt': inp_event_cnt,  # 2xHxW, 0 for positive, 1 for negtive
            'gt_cnt': gt_event_cnt,  # 2xkHxkW, 0 for positive, 1 for negtive
        }

        return item

    def __len__(self):

        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 < self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))

        return idx0, idx1

    def get_gt_event_indices(self, index):
        """
        Get start and end indices of gt events at index
        """
        gt_idx0, gt_idx1 = self.gt_event_indices[index]
        if not (gt_idx0 >= 0 and gt_idx1 < self.num_gt_events):
            raise Exception(
                "WARNING: Gt event indices {},{} out of bounds 0,{}".format(gt_idx0, gt_idx1, self.num_gt_events))

        return gt_idx0, gt_idx1

    def get_gt_event_indices_time(self, idx0, idx1):
        """
        Get start and end indices of gt events using idx0 and idx1 of input events based on time.
        """
        gt_t0, gt_tk = self.h5_file[f'{self.inp_prex}_events/ts'][idx0], self.h5_file[f'{self.inp_prex}_events/ts'][
            idx1]
        gt_idx0 = self.binary_search_h5_dset(self.h5_file[f'{self.gt_prex}_events/ts'][:], gt_t0)
        gt_idx1 = self.binary_search_h5_dset(self.h5_file[f'{self.gt_prex}_events/ts'][:], gt_tk)
        if gt_idx0 < 0:
            gt_idx0 = 0
        if gt_idx1 > self.num_gt_events - 1:
            gt_idx1 = self.num_gt_events - 1

        if not (gt_idx0 >= 0 and gt_idx1 < self.num_gt_events):
            raise Exception(
                "WARNING: GT event indices {},{} out of bounds 0,{}".format(gt_idx0, gt_idx1, self.num_gt_events))

        return gt_idx0, gt_idx1

    def get_gt_event_indices_num(self, k_indices):
        """
        Get start and end indices of gt events using idx0 and idx1 of input events based on numbers.
        """

        num_events = k_indices[0][1] - k_indices[0][0]
        num_gt_events = self.scale ** 2 * num_events

        gt_t0_list = []
        gt_file = self.h5_file[f'{self.inp_prex}_events/ts']
        for i in range(len(k_indices)):
            gt_t0_list.append(gt_file[k_indices[i][0]])

        gt_idx0_list = self.binary_search_h5_dset(self.h5_file[f'{self.gt_prex}_events/ts'][:], gt_t0_list)

        gt_k_indices = []
        for i in range(len(gt_idx0_list)):
            gt_idx0 = gt_idx0_list[i]
            gt_idx1 = gt_idx0 + num_gt_events

            if gt_idx0 < 0:
                gt_idx0 = 0
                gt_idx1 = gt_idx0 + num_gt_events
            if gt_idx1 > self.num_gt_events - 1:
                gt_idx1 = self.num_gt_events - 1
                gt_idx0 = gt_idx1 - num_gt_events
            gt_k_indices.append([gt_idx0, gt_idx1])

        return gt_k_indices

    def get_gt_frame(self, event_idx0, event_idx1):
        ref_idx = int((event_idx0 + event_idx1) // 2)
        event_ts = self.h5_file[f'{self.inp_prex}_events/ts'][ref_idx]
        gt_img_idx = self.binary_search_h5_dset(self.gt_frame_ts, event_ts)

        if gt_img_idx >= len(self.gt_frame_ts):
            gt_img_idx = len(self.gt_frame_ts) - 1
        if gt_img_idx < 0:
            gt_img_idx = 0

        return self.h5_file['ori_images/image{:09d}'.format(gt_img_idx)][:]

    def get_frame(self, index):
        return self.h5_file['ori_images']['image{:09d}'.format(index)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file[f'{self.inp_prex}_events/xs'][idx0:idx1]
        ys = self.h5_file[f'{self.inp_prex}_events/ys'][idx0:idx1]
        ts = self.h5_file[f'{self.inp_prex}_events/ts'][idx0:idx1]
        ps = self.h5_file[f'{self.inp_prex}_events/ps'][idx0:idx1]

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]),
                              axis=0)

    def get_gt_events(self, idx0, idx1):
        xs = self.h5_file[f'{self.gt_prex}_events/xs'][idx0:idx1]
        ys = self.h5_file[f'{self.gt_prex}_events/ys'][idx0:idx1]
        ts = self.h5_file[f'{self.gt_prex}_events/ts'][idx0:idx1]
        ps = self.h5_file[f'{self.gt_prex}_events/ps'][idx0:idx1]

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]),
                              axis=0)

    def create_normalized_events(self, events, sensor_resolution):
        """
        events: torch.tensor, 4xN [x, y, t, p]

        return: normalized events: torch.tensor, 4xN [x, y, t, p]
        """
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]
        xs, ys = xs / sensor_resolution[1], ys / sensor_resolution[0]
        norm_events = torch.stack([xs, ys, ts, ps]).float()

        return norm_events

    def create_scaled_encoding(self, normalized_events, sensor_resolution, mode):
        """
        normalized events: torch.tensor, 4xN [x, y, t, p]

        return: scaled data
        """
        xs, ys, ts, ps = normalized_events[0], normalized_events[1], normalized_events[2], normalized_events[3]
        if mode == 'cnt':
            scaled_data = events_to_channels(xs * sensor_resolution[1], ys * sensor_resolution[0], ps,
                                             sensor_size=sensor_resolution)
        elif mode == 'stack':
            scaled_data = events_to_stack_no_polarity(xs * sensor_resolution[1], ys * sensor_resolution[0], ts, ps,
                                                      B=self.time_bins, sensor_size=sensor_resolution)
        elif mode == 'events':
            scaled_data = torch.stack([(xs * sensor_resolution[1]).long(), (ys * sensor_resolution[0]).long(), ts, ps],
                                      dim=0)
        else:
            raise Exception(f'mode: {mode} is NOT supported!')

        return scaled_data

    def create_unsupervised_data(self, normalized_events):
        """
        normalized events: torch.tensor, 4xN [x, y, t, p]

        return: scaled data
        """
        xs, ys, ts, ps = normalized_events[0], normalized_events[1], normalized_events[2], normalized_events[3]
        inp_down_events = torch.stack(
            [(xs * self.inp_down_sensor_resolution[1]).long(), (ys * self.inp_down_sensor_resolution[0]).long(), ts,
             ps], dim=0)
        inp_down_normalized_events = self.create_normalized_events(inp_down_events, self.inp_down_sensor_resolution)
        # inp_down_cnt = self.create_scaled_encoding(inp_down_normalized_events, self.inp_down_sensor_resolution, mode='cnt') // self.scale**2
        inp_down_cnt = torch.div(
            self.create_scaled_encoding(inp_down_normalized_events, self.inp_down_sensor_resolution, mode='cnt'),
            self.scale ** 2, rounding_mode='trunc')
        # inp_down_scaled_cnt = self.create_scaled_encoding(inp_down_normalized_events, self.inp_sensor_resolution, mode='cnt') // self.scale**2
        inp_down_scaled_cnt = torch.div(
            self.create_scaled_encoding(inp_down_normalized_events, self.inp_sensor_resolution, mode='cnt'),
            self.scale ** 2, rounding_mode='trunc')

        return inp_down_cnt, inp_down_scaled_cnt


    def create_custom_data(self, inp_cnt, inp_scaled_cnt, inp_down_cnt, inp_down_scaled_cnt, gt_cnt):
        inp_custom_cnt = func.interpolate(inp_cnt.unsqueeze(0), size=self.custom_resolution, mode='bicubic',
                                          align_corners=False).squeeze(0)
        inp_custom_scaled_cnt = func.interpolate(inp_scaled_cnt.unsqueeze(0),
                                                 size=[i * self.scale for i in self.custom_resolution], mode='bicubic',
                                                 align_corners=False).squeeze(0)
        inp_custom_down_cnt = func.interpolate(inp_down_cnt.unsqueeze(0),
                                               size=[round(i / self.scale) for i in self.custom_resolution],
                                               mode='bicubic', align_corners=False).squeeze(0)
        inp_custom_down_scaled_cnt = func.interpolate(inp_down_scaled_cnt.unsqueeze(0), size=self.custom_resolution,
                                                      mode='bicubic', align_corners=False).squeeze(0)
        gt_custom_cnt = func.interpolate(gt_cnt.unsqueeze(0), size=[i * self.scale for i in self.custom_resolution],
                                         mode='bicubic', align_corners=False).squeeze(0)

        return inp_custom_cnt.round(), inp_custom_scaled_cnt.round(), inp_custom_down_cnt.round(), inp_custom_down_scaled_cnt.round(), gt_custom_cnt.round()

    def create_voxel_encoding(self, events, sensor_resolution):
        """
        events: torch.tensor, 4xN [x, y, t, p]

        return: voxel: torch.tensor, B x H x W
        """
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]

        # return events_to_voxel_torch(xs, ys, ts, ps, B=self.time_bins, sensor_size=sensor_resolution)
        return events_to_voxel(xs, ys, ts, ps, num_bins=self.time_bins, sensor_size=sensor_resolution)

    def create_stack_encoding(self, events, sensor_resolution):
        """
        events: torch.tensor, 4xN [x, y, t, p]

        return: stack: torch.tensor, B x H x W
        """
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]

        return events_to_stack_no_polarity(xs, ys, ts, ps, B=self.time_bins, sensor_size=sensor_resolution)

    def create_cnt_encoding(self, events, sensor_resolution):
        """
        events: torch.tensor, 4xN [x, y, t, p]

        return: count: torch.tensor, 2 x H x W
        """
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]

        return events_to_channels(xs, ys, ps, sensor_size=sensor_resolution)

    def create_hot_mask(self, events, sensor_resolution):
        """
        Creates a one channel tensor that can act as mask to remove pixel with high event rate.
        events: torch.tensor, 4xN [x, y, t, p]

        return: [H x W] binary mask
        """
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]

        hot_update = events_to_mask(xs, ys, ps, sensor_size=sensor_resolution)
        self.hot_events += hot_update
        self.hot_idx += 1
        event_rate = self.hot_events / self.hot_idx

        return get_hot_event_mask(
            event_rate,
            self.hot_idx,
            max_px=self.config["hot_filter"]["max_px"],
            min_obvs=self.config["hot_filter"]["min_obvs"],
            max_rate=self.config["hot_filter"]["max_rate"],
        )

    @staticmethod
    def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """
        return events_polarity_mask(ps)

    def augment_event(self, events, sensor_resolution, seed):
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]
        seed_H, seed_W, seed_P = seed, seed + 1, seed + 2

        for i, mechanism in enumerate(self.config['data_augment']['augment']):
            if mechanism == 'Horizontal':
                random.seed(seed_H)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    xs = sensor_resolution[1] - 1 - xs
            elif mechanism == 'Vertical':
                random.seed(seed_W)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    ys = sensor_resolution[0] - 1 - ys
            elif mechanism == 'Polarity':
                random.seed(seed_P)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    ps = ps * -1

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]),
                              axis=0)

    def augment_frame(self, img, seed):
        seed_H, seed_W = seed, seed + 1

        for i, mechanism in enumerate(self.config['data_augment']['augment']):
            if mechanism == 'Horizontal':
                random.seed(seed_H)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    img = np.flip(img, 1)
            elif mechanism == 'Vertical':
                random.seed(seed_W)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    img = np.flip(img, 0)

        return img

    @staticmethod
    def add_hot_pixels_to_voxel(voxel, hot_pixel_std=1.0, hot_pixel_fraction=0.001):
        num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
        x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
        y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
        for i in range(num_hot_pixels):
            voxel[..., :, y[i], x[i]] += random.gauss(0, hot_pixel_std)

    @staticmethod
    def add_noise_cnt(size, seed, noise_std=1.0, noise_fraction=0.1):
        torch.manual_seed(seed)
        noise = torch.abs(noise_std * torch.randn(size))  # mean = 0, std = noise_std
        if noise_fraction < 1.0:
            mask = torch.rand(size) >= noise_fraction
            noise.masked_fill_(mask, 0)

        return noise

    @staticmethod
    def add_noise_stack(size, seed, noise_std=1.0, noise_fraction=0.1):
        torch.manual_seed(seed)
        noise = noise_std * torch.randn(size)  # mean = 0, std = noise_std
        if noise_fraction < 1.0:
            mask = torch.rand(size) >= noise_fraction
            noise.masked_fill_(mask, 0)

        return noise

    @staticmethod
    def add_noise_event(window, sensor_size, seed, noise_level=0.01):
        torch.manual_seed(seed)
        noise_num = int(window * noise_level)  # * (1 + noise_std * torch.randn(1).abs().max().item())
        noise_tmp = torch.rand([4, noise_num])
        x = (noise_tmp[[0], :] * sensor_size[1]).int()
        y = (noise_tmp[[1], :] * sensor_size[0]).int()
        t = torch.ones_like(y)
        p = (noise_tmp[[3], :] * 2).int() * 2 - 1
        noise = torch.cat([x, y, t, p], dim=0)

        return noise


class SequenceDataset(Dataset):
    def __init__(self, h5_file_path, config):
        super().__init__()

        self.config = config
        self.L = config['sequence']['sequence_length']
        step_size = config['sequence']['step_size']
        self.step_size = step_size if step_size is not None else self.L
        self.proba_pause_when_running = config['sequence']['pause']['proba_pause_when_running']
        self.proba_pause_when_paused = config['sequence']['pause']['proba_pause_when_paused']

        assert (self.L > 0)
        assert (self.step_size > 0)

        self.dataset = H5Dataset(h5_file_path, config)
        if self.L >= self.dataset.length:
            print(
                f'Set sequence: {h5_file_path} length {self.L} is bigger than the max length of dataset {self.dataset.length}')
            self.length = 1
            self.L = self.dataset.length
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1

        self.gt_sensor_resolution = self.dataset.gt_sensor_resolution
        self.inp_sensor_resolution = self.dataset.inp_sensor_resolution

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        assert (i >= 0)
        assert (i < self.length)

        seed = random.randint(0, 2 ** 32)

        sequence = []
        k = 0
        j = i * self.step_size
        item = self.dataset.__getitem__(j, seed=seed)
        sequence.append(item)

        paused = False
        for n in range(self.L - 1):

            if self.config['sequence']['pause']['enabled']:
                u = random.random()
                if paused:
                    probability_pause = self.proba_pause_when_paused
                else:
                    probability_pause = self.proba_pause_when_running
                paused = (u < probability_pause)

            if paused:
                # add a tensor filled with zeros, paired with the last item
                # do not increase the counter
                item = self.dataset.__getitem__(j + k, Pause=True, seed=seed)
                sequence.append(item)
            else:
                # normal case: append the next item to the list
                k += 1
                item = self.dataset.__getitem__(j + k, seed=seed)
                sequence.append(item)

        return sequence


