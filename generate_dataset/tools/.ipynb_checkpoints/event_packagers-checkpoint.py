from abc import ABCMeta, abstractmethod
from shutil import Error
import h5py
import cv2 as cv
import numpy as np

class packager():

    __metaclass__ = ABCMeta

    def __init__(self, name, output_path, max_buffer_size=1000000):
        self.name = name
        self.output_path = output_path
        self.max_buffer_size = max_buffer_size

    @abstractmethod
    def package_events(self, xs, ys, ts, ps):
        pass

    @abstractmethod
    def package_image(self, frame, timestamp):
        pass

    @abstractmethod
    def package_flow(self, flow, timestamp):
        pass

    @abstractmethod
    def add_metadata(self, num_events, num_pos, num_neg,
            duration, t0, tk, num_imgs, num_flow):
        pass

    @abstractmethod
    def set_data_available(self, num_images, num_flow):
        pass

class hdf5_packager(packager):
    """
    This class packages data to hdf5 files
    """
    def __init__(self, output_path, max_buffer_size=1000000):
        packager.__init__(self, 'hdf5', output_path, max_buffer_size)
        print("CREATING FILE IN {}".format(output_path))
        self.events_file = h5py.File(output_path, 'w')
        self.event_xs = self.events_file.create_dataset("events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.event_ys = self.events_file.create_dataset("events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.event_ts = self.events_file.create_dataset("events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.event_ps = self.events_file.create_dataset("events/ps", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

    def append_to_dataset(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data):] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append_to_dataset(self.event_xs, xs)
        self.append_to_dataset(self.event_ys, ys)
        self.append_to_dataset(self.event_ts, ts)
        self.append_to_dataset(self.event_ps, ps)

    def package_image(self, image, timestamp, img_idx):
        image_dset = self.events_file.create_dataset("images/image{:09d}".format(img_idx),
                data=image, dtype=np.dtype(np.uint8))
        image_dset.attrs['size'] = image.shape
        image_dset.attrs['timestamp'] = timestamp
        image_dset.attrs['type'] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr" 

    def package_flow(self, flow_image, timestamp, flow_idx):
        flow_dset = self.events_file.create_dataset("flow/flow{:09d}".format(flow_idx),
                data=flow_image, dtype=np.dtype(np.float32))
        flow_dset.attrs['size'] = flow_image.shape
        flow_dset.attrs['timestamp'] = timestamp

    def add_event_indices(self):
        datatypes = ['images', 'flow']
        for datatype in datatypes:
            if datatype in self.events_file.keys():
                s = 0
                added = 0
                ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                for image in self.events_file[datatype]:
                    img_ts = self.events_file[datatype][image].attrs['timestamp']
                    event_idx = np.searchsorted(ts, img_ts)
                    if event_idx == len(ts):
                        added += len(ts)
                        s += self.max_buffer_size
                        ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                        event_idx = np.searchsorted(ts, img_ts)
                    event_idx = max(0, event_idx-1)
                    self.events_file[datatype][image].attrs['event_idx'] = event_idx + added

    def add_metadata(self, num_pos, num_neg,
            duration, t0, tk, num_imgs, num_flow, sensor_size):
        self.events_file.attrs['num_events'] = num_pos+num_neg
        self.events_file.attrs['num_pos'] = num_pos
        self.events_file.attrs['num_neg'] = num_neg
        self.events_file.attrs['duration'] = tk-t0
        self.events_file.attrs['t0'] = t0
        self.events_file.attrs['tk'] = tk
        self.events_file.attrs['num_imgs'] = num_imgs
        self.events_file.attrs['num_flow'] = num_flow
        self.events_file.attrs['sensor_resolution'] = sensor_size
        self.add_event_indices()

    def set_data_available(self, num_images, num_flow):
        if num_images > 0:
            self.image_dset = self.events_file.create_group("images")
            self.image_dset.attrs['num_images'] = num_images
        if num_flow > 0:
            self.flow_dset = self.events_file.create_group("flow")
            self.flow_dset.attrs['num_images'] = num_flow

    def add_data(self, sensor_size, num_imgs):
        self.events_file.attrs['sensor_resolution'] = sensor_size
        self.events_file.attrs['num_imgs'] = num_imgs


class hdf5_event_packager(packager):
    """
    This class packages data to hdf5 files
    """
    def __init__(self, output_path, max_buffer_size=1000000):
        packager.__init__(self, 'hdf5', output_path, max_buffer_size)
        print("CREATING FILE IN {}".format(output_path))
        self.events_file = h5py.File(output_path, 'w')

        self.ori_event_xs = self.events_file.create_dataset("ori_events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.ori_event_ys = self.events_file.create_dataset("ori_events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.ori_event_ts = self.events_file.create_dataset("ori_events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.ori_event_ps = self.events_file.create_dataset("ori_events/ps", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

        self.down2_event_xs = self.events_file.create_dataset("down2_events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down2_event_ys = self.events_file.create_dataset("down2_events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down2_event_ts = self.events_file.create_dataset("down2_events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.down2_event_ps = self.events_file.create_dataset("down2_events/ps", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

        self.down4_event_xs = self.events_file.create_dataset("down4_events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down4_event_ys = self.events_file.create_dataset("down4_events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down4_event_ts = self.events_file.create_dataset("down4_events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.down4_event_ps = self.events_file.create_dataset("down4_events/ps", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

        self.down8_event_xs = self.events_file.create_dataset("down8_events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down8_event_ys = self.events_file.create_dataset("down8_events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down8_event_ts = self.events_file.create_dataset("down8_events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.down8_event_ps = self.events_file.create_dataset("down8_events/ps", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

        # self.down8_real_event_xs = self.events_file.create_dataset("down8_real_events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        # self.down8_real_event_ys = self.events_file.create_dataset("down8_real_events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        # self.down8_real_event_ts = self.events_file.create_dataset("down8_real_events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        # self.down8_real_event_ps = self.events_file.create_dataset("down8_real_events/ps", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

        self.down16_event_xs = self.events_file.create_dataset("down16_events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down16_event_ys = self.events_file.create_dataset("down16_events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.down16_event_ts = self.events_file.create_dataset("down16_events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.down16_event_ps = self.events_file.create_dataset("down16_events/ps", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

    def append_to_dataset(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data):] = data[:]

    def package_image(self, type, image, timestamp, img_idx):
        image_dset = self.events_file.create_dataset("{}_images/image{:09d}".format(type, img_idx),
                data=image, dtype=np.dtype(np.uint8))
        image_dset.attrs['size'] = image.shape
        image_dset.attrs['timestamp'] = timestamp
        image_dset.attrs['type'] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr"

    def package_events(self, type, xs, ys, ts, ps):
        if type == 'ori':
            self.append_to_dataset(self.ori_event_xs, xs)
            self.append_to_dataset(self.ori_event_ys, ys)
            self.append_to_dataset(self.ori_event_ts, ts)
            self.append_to_dataset(self.ori_event_ps, ps)
        elif type == 'down2':
            self.append_to_dataset(self.down2_event_xs, xs)
            self.append_to_dataset(self.down2_event_ys, ys)
            self.append_to_dataset(self.down2_event_ts, ts)
            self.append_to_dataset(self.down2_event_ps, ps)
        elif type == 'down4':
            self.append_to_dataset(self.down4_event_xs, xs)
            self.append_to_dataset(self.down4_event_ys, ys)
            self.append_to_dataset(self.down4_event_ts, ts)
            self.append_to_dataset(self.down4_event_ps, ps)
        elif type == 'down8':
            self.append_to_dataset(self.down8_event_xs, xs)
            self.append_to_dataset(self.down8_event_ys, ys)
            self.append_to_dataset(self.down8_event_ts, ts)
            self.append_to_dataset(self.down8_event_ps, ps)
        elif type == 'down16':
            self.append_to_dataset(self.down16_event_xs, xs)
            self.append_to_dataset(self.down16_event_ys, ys)
            self.append_to_dataset(self.down16_event_ts, ts)
            self.append_to_dataset(self.down16_event_ps, ps)
        else:
            raise Exception(f'Not support {type}')

    def add_event_indices(self):
        datatypes = ['ori_images']
        for datatype in datatypes:
            if datatype in self.events_file.keys():
                for prex in ['ori', 'down2', 'down4', 'down8', 'down16']:
                    s = 0
                    added = 0
                    for image in self.events_file[datatype]:
                        img_ts = self.events_file[datatype][image].attrs['timestamp']
                        while True:
                            ts = self.events_file[f"{prex}_events/ts"][s:s+self.max_buffer_size]
                            if len(ts) == 0:
                                break
                            event_idx = np.searchsorted(ts, img_ts)
                            if event_idx < len(ts):
                                break
                            added += len(ts)
                            s += self.max_buffer_size
                        event_idx = max(0, event_idx-1)
                        final_idx = min(self.events_file[f"{prex}_events/ts"].shape[0]-1, event_idx + added)
                        self.events_file[datatype][image].attrs[f'{prex}_event_idx'] = final_idx

    def add_data(self, sensor_size):
        self.events_file.attrs['sensor_resolution'] = sensor_size

