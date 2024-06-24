import os
import cv2
import numpy as np
import torch
import h5py
import argparse


class Visualization:
    def __init__(self, px=400, color_scheme='green_red', eval_id=-1):
        self.img_idx = 0
        self.px = px
        self.color_scheme = color_scheme  # gray / blue_red / green_red

    def plot_event(self, event_cnt, is_save, name='events_img'):

        event_img = (self.events_to_image(event_cnt, self.color_scheme)*255).astype(np.uint8)

        cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{name}", int(self.px), int(self.px))
        cv2.imshow(f"{name}", event_img)

        if is_save:
            filename = '/tmp/event_img.png'
            cv2.imwrite(filename, event_img)

    def plot_frame(self, frame, name='frame'):

        cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{name}", int(self.px), int(self.px))
        cv2.imshow(f"{name}", frame)
    
    @staticmethod
    def events_to_image(inp_events, color_scheme="green_red"):
        """
        Visualize the input events.
        :param inp_events: [H x W x 2] per-pixel and per-polarity event count, numpy.narray
        :param color_scheme: green_red/gray/blue_red
        :return event_image: [H x W x 3] color-coded event image, range: [0, 1]
        """
        assert color_scheme in ['green_red', 'gray', 'blue_red'], f'Not support {color_scheme}'

        pos = inp_events[:, :, 0]
        neg = inp_events[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((inp_events.shape[0], inp_events.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        elif color_scheme == "blue_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 1][mask_pos] = 0
            event_image[:, :, 0][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 1][mask_neg] = 0
            event_image[:, :, 0][mask_neg * mask_not_pos] = 0

        return event_image

    def events_to_channels(self, xs, ys, ps, sensor_size=(180, 240)):
        """
        Generate a two-channel event image containing event counters.
        """
    
        assert len(xs) == len(ys) and len(ys) == len(ps)

        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ps = torch.from_numpy(ps)
    
        mask_pos = ps.clone()
        mask_neg = ps.clone()
        mask_pos[ps < 0] = 0
        mask_neg[ps > 0] = 0
    
        pos_cnt = self.events2image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
        neg_cnt = self.events2image(xs, ys, ps * mask_neg, sensor_size=sensor_size)
    
        return torch.stack([pos_cnt, neg_cnt], dim=-1).numpy()

    @staticmethod
    def events2image(xs, ys, ps, sensor_size=(180, 240)):
        """
        Accumulate events into an image.
        """

        device = xs.device
        img_size = list(sensor_size)
        img = torch.zeros(img_size).to(device)

        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)

        return img


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_file_path', required=True)
    parser.add_argument('--idx', type=int, default=3)
    flags = parser.parse_args()

    return flags


if __name__ == '__main__':
    """
    usage for event visualization:
        python utils/event_visual_example.py --h5_file_path path/to/event.h5
    """

    flags = get_flags()

    h5_file_path = flags.h5_file_path
    idx = flags.idx

    assert os.path.isfile(h5_file_path)
    assert idx < 7 and idx > 0

    vis = Visualization()

    event_h5 = h5py.File(h5_file_path, 'r')

    sensor_resolution = event_h5.attrs['sensor_resolution']

    frame0_h5 = event_h5['images']['image{:09d}'.format(idx-1)]
    frame1_h5 = event_h5['images']['image{:09d}'.format(idx)]
    frame0 = frame0_h5[:]
    frame1 = frame1_h5[:]

    events_idx = [frame0_h5.attrs['event_idx'], frame1_h5.attrs['event_idx']]

    xs = event_h5['events/xs'][events_idx[0]:events_idx[1]].astype(np.float32)
    ys = event_h5['events/ys'][events_idx[0]:events_idx[1]].astype(np.float32)
    ts = event_h5['events/ts'][events_idx[0]:events_idx[1]].astype(np.float32)
    ps = event_h5['events/ps'][events_idx[0]:events_idx[1]].astype(np.float32)

    event_cnt = vis.events_to_channels(xs, ys, ps, sensor_resolution)

    vis.plot_event(event_cnt, is_save=True) # set is_save to True to save event img
    vis.plot_frame(frame0)

    cv2.waitKey()
    cv2.destroyAllWindows()

