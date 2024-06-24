import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('seaborn-whitegrid')
import os
import matplotlib.animation as animation
from tqdm import tqdm
import open3d as o3d
# local modules
from myutils.vis_events.visualization import *

from dataloader.h5dataloader import InferenceHDF5DataLoader
from dataloader.encodings import *


def check(input, resolution):
    def convert_polarity(ps):
        """
        ps: torch.tensor, [N], -1~+1
        """
        pos_mask = ps > 0
        neg_mask = ps < 0
        ps[pos_mask] = 1
        ps[neg_mask] = -1
        return ps

    input = input.detach().cpu().clone()

    vis = event_visualisation()
    H, W = resolution[0], resolution[1]
    cnt = events_to_channels(input[0, :, 0]*W, input[0, :, 1]*H, convert_polarity(input[0, :, 3]), resolution)
    vis.plot_event_cnt(cnt.numpy().transpose(1, 2, 0), is_save=False, is_black_background=True)

    plt.show()


def show_event_cloud(sparse_points):
    def polarity2color(polarity):
        pos = polarity == 1
        neg = polarity == -1
        color = np.zeros((polarity.shape[0], 3))
        color[:, 1][pos] = 1
        color[:, 0][neg] = 1
        return color

    batch_coords, batch_feats = sparse_points.decomposed_coordinates_and_features
    points = batch_coords[0].cpu().numpy()
    polarity = batch_feats[0][:, -1].cpu().detach().numpy()
    color = polarity2color(polarity)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud('/data/output/inferout/point.ply', pcd)

    return pcd

class event_visualisation():
    def plot_data(self, data: np.ndarray, path, is_save, DPI=300, cmap=None):
        H, W = data.shape[0:2]
        fig = plt.figure(figsize=(W / float(DPI), H / float(DPI))) if is_save else plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off') # remove white border
        ax.set_xticks([]);ax.set_yticks([])
        if cmap is not None:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        if is_save:
            assert path is not None
            fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0)

    def plot_frame(self, frame, is_save, path=None, cmap='gray'):
        """
        frame: np.ndarray, HxW
        """
        self.plot_data(frame, path, is_save, cmap=cmap)

    def plot_event_stack(self, event_stack, is_save, path=None, DPI=300):
        """
        event_stack: np.ndarray, HxWxC

        blue for positive, red for negative
        """
        
        time_bins = event_stack.shape[-1]
        h = int(np.sqrt(time_bins))
        w = time_bins // h
        assert h * w == time_bins

        v_min = -10 #event_stack.min() * 0.9
        v_max = -v_min

        fig = plt.figure(figsize=(10,10))
        
        grid = ImageGrid(fig, 111,
                          nrows_ncols=(h,w),
                          axes_pad=0.15,
                          share_all=True,
                          cbar_location="right",
                          cbar_mode="single",
                          cbar_size="3%",
                          cbar_pad=0.15,
                         )
        
        # Add data to image grid
        for i, ax in enumerate(grid):
            im = ax.imshow(event_stack[..., i], cmap='RdBu'
                                        , vmin=v_min, vmax=v_max
                                        )
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Colorbar
        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)

        if is_save:
            plt.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0)

        return fig

    def plot_event_cnt(self, event_cnt, is_save, path=None, color_scheme="blue_red", use_opencv=False, is_black_background=False, is_norm=True):
        """
        event_cnt: np.ndarray, HxWx2, 0 for positive, 1 for negative

        'gray': white for positive, black for negative
        'green_red': green for positive, red for negative
        'blue_red': blue for positive, red for negative
        """
        assert color_scheme in ['green_red', 'gray', 'blue_red'], f'Not support {color_scheme}'

        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if is_norm:
            if pos_min != max:
                pos = (pos - pos_min) / (max - pos_min)
            if neg_min != max:
                neg = (neg - neg_min) / (max - neg_min)
        else:
            mask_pos_nonzero = pos != 0
            mask_neg_nonzero = neg != 0
            mask_posnonnorm = (pos >= neg) * mask_pos_nonzero
            mask_negnonnorm = (pos < neg) * mask_neg_nonzero
            pos[mask_posnonnorm] = 1
            neg[mask_posnonnorm] = 0
            neg[mask_negnonnorm] = 1
            pos[mask_negnonnorm] = 0

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            if is_black_background:
                event_image *= 0
                event_image[:, :, 0][mask_pos] = 0
                event_image[:, :, 1][mask_pos] = pos[mask_pos]
                event_image[:, :, 2][mask_pos * mask_not_neg] = 0
                event_image[:, :, 2][mask_neg] = neg[mask_neg]
                event_image[:, :, 0][mask_neg] = 0
                event_image[:, :, 1][mask_neg * mask_not_pos] = 0
            else:
                # only pos
                event_image[:, :, 0][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                event_image[:, :, 1][mask_pos * mask_not_neg] = 1
                event_image[:, :, 2][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                # only neg
                event_image[:, :, 2][mask_neg * mask_not_pos] = 1
                event_image[:, :, 0][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                event_image[:, :, 1][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                ######### pos + neg
                mask_posoverneg = pos >= neg
                mask_negoverpos = pos < neg
                # pos >= neg
                event_image[:, :, 0][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[mask_pos * mask_neg * mask_posoverneg]
                event_image[:, :, 1][mask_pos * mask_neg * mask_posoverneg] = 1
                event_image[:, :, 2][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[mask_pos * mask_neg * mask_posoverneg]
                # pos < neg
                event_image[:, :, 2][mask_pos * mask_neg * mask_negoverpos] = 1
                event_image[:, :, 0][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[mask_pos * mask_neg * mask_negoverpos]
                event_image[:, :, 1][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[mask_pos * mask_neg * mask_negoverpos]

        elif color_scheme == "blue_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)

            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            if is_black_background:
                event_image *= 0
                event_image[:, :, 1][mask_pos] = 0
                event_image[:, :, 0][mask_pos] = pos[mask_pos]
                event_image[:, :, 2][mask_pos * mask_not_neg] = 0
                event_image[:, :, 2][mask_neg] = neg[mask_neg]
                event_image[:, :, 1][mask_neg] = 0
                event_image[:, :, 0][mask_neg * mask_not_pos] = 0
            else:
                # only pos
                event_image[:, :, 0][mask_pos * mask_not_neg] = 1 
                event_image[:, :, 1][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                event_image[:, :, 2][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                # only neg
                event_image[:, :, 2][mask_neg * mask_not_pos] = 1
                event_image[:, :, 0][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                event_image[:, :, 1][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                ######### pos + neg
                mask_posoverneg = pos >= neg
                mask_negoverpos = pos < neg
                # pos >= neg
                event_image[:, :, 0][mask_pos * mask_neg * mask_posoverneg] = 1 
                event_image[:, :, 1][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[mask_pos * mask_neg * mask_posoverneg]
                event_image[:, :, 2][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[mask_pos * mask_neg * mask_posoverneg]
                # pos < neg
                event_image[:, :, 2][mask_pos * mask_neg * mask_negoverpos] = 1
                event_image[:, :, 0][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[mask_pos * mask_neg * mask_negoverpos]
                event_image[:, :, 1][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[mask_pos * mask_neg * mask_negoverpos]

        event_image = (event_image * 255).astype(np.uint8)
        if not use_opencv:
            event_image = cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB)

        self.plot_data(event_image, path, is_save)

        return event_image

    def plot_event_stack_cnt(self, event_cnt, is_save, path=None, color_scheme="blue_red", use_opencv=False,
                       is_black_background=False, is_norm=True):
        """
        event_cnt: np.ndarray, HxWx2, 0 for positive, 1 for negative

        'gray': white for positive, black for negative
        'green_red': green for positive, red for negative
        'blue_red': blue for positive, red for negative
        """
        assert color_scheme in ['green_red', 'gray', 'blue_red'], f'Not support {color_scheme}'

        pos = np.zeros((event_cnt.shape[0], event_cnt.shape[1]))
        neg = np.zeros((event_cnt.shape[0], event_cnt.shape[1]))
        for i in range(event_cnt.shape[0]):
            for j in range(event_cnt.shape[1]):
                if event_cnt[i,j,0] > 0:
                    pos[i,j] = event_cnt[i,j,0]
                elif event_cnt[i,j,0] < 0:
                    neg[i,j] = event_cnt[i,j,0]

        neg = -neg
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if is_norm:
            if pos_min != max:
                pos = (pos - pos_min) / (max - pos_min)
            if neg_min != max:
                neg = (neg - neg_min) / (neg_max - neg_min)
        else:
            mask_pos_nonzero = pos != 0
            mask_neg_nonzero = neg != 0
            mask_posnonnorm = (pos >= neg) * mask_pos_nonzero
            mask_negnonnorm = (pos < neg) * mask_neg_nonzero
            pos[mask_posnonnorm] = 1
            neg[mask_posnonnorm] = 0
            neg[mask_negnonnorm] = 1
            pos[mask_negnonnorm] = 0

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)

            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            if is_black_background:
                event_image *= 0
                event_image[:, :, 0][mask_pos] = 0
                event_image[:, :, 1][mask_pos] = pos[mask_pos]
                event_image[:, :, 2][mask_pos * mask_not_neg] = 0
                event_image[:, :, 2][mask_neg] = neg[mask_neg]
                event_image[:, :, 0][mask_neg] = 0
                event_image[:, :, 1][mask_neg * mask_not_pos] = 0
            else:
                # only pos
                event_image[:, :, 0][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                event_image[:, :, 1][mask_pos * mask_not_neg] = 1
                event_image[:, :, 2][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                # only neg
                event_image[:, :, 2][mask_neg * mask_not_pos] = 1
                event_image[:, :, 0][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                event_image[:, :, 1][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                ######### pos + neg
                mask_posoverneg = pos >= neg
                mask_negoverpos = pos < neg
                # pos >= neg
                event_image[:, :, 0][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[
                    mask_pos * mask_neg * mask_posoverneg]
                event_image[:, :, 1][mask_pos * mask_neg * mask_posoverneg] = 1
                event_image[:, :, 2][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[
                    mask_pos * mask_neg * mask_posoverneg]
                # pos < neg
                event_image[:, :, 2][mask_pos * mask_neg * mask_negoverpos] = 1
                event_image[:, :, 0][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[
                    mask_pos * mask_neg * mask_negoverpos]
                event_image[:, :, 1][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[
                    mask_pos * mask_neg * mask_negoverpos]

        elif color_scheme == "blue_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)

            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            if is_black_background:
                event_image *= 0
                event_image[:, :, 1][mask_pos] = 0
                event_image[:, :, 0][mask_pos] = pos[mask_pos]
                event_image[:, :, 2][mask_pos * mask_not_neg] = 0
                event_image[:, :, 2][mask_neg] = neg[mask_neg]
                event_image[:, :, 1][mask_neg] = 0
                event_image[:, :, 0][mask_neg * mask_not_pos] = 0
            else:
                # only pos
                event_image[:, :, 0][mask_pos * mask_not_neg] = 1
                event_image[:, :, 1][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                event_image[:, :, 2][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
                # # only neg
                event_image[:, :, 2][mask_neg * mask_not_pos] = 1
                event_image[:, :, 0][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                event_image[:, :, 1][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
                ######### pos + neg
                mask_posoverneg = pos > neg
                mask_negoverpos = pos < neg
                # pos >= neg
                event_image[:, :, 0][mask_pos * mask_neg * mask_posoverneg] = 1
                event_image[:, :, 1][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[
                    mask_pos * mask_neg * mask_posoverneg]
                event_image[:, :, 2][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[
                    mask_pos * mask_neg * mask_posoverneg]
                # pos < neg
                event_image[:, :, 2][mask_pos * mask_neg * mask_negoverpos] = 1
                event_image[:, :, 0][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[
                    mask_pos * mask_neg * mask_negoverpos]
                event_image[:, :, 1][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[
                    mask_pos * mask_neg * mask_negoverpos]

        event_image = (event_image * 255).astype(np.uint8)
        if not use_opencv:
            event_image = cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB)

        self.plot_data(event_image, path, is_save)

        return event_image

    def plot_event_img(self, event_list, resolution, is_save, path=None):
        """
        event_list: np.ndarray, Nx4, [x, y, t, p], p:[-1, 1]
        resolution: list, [H,W]

        blue for positive, red for negative
        """
        x, y, p = event_list[:, 0], event_list[:, 1], event_list[:, 3]
        H, W = resolution[0], resolution[1]

        assert x.size == y.size == p.size
        assert H > 0
        assert W > 0

        x = x.astype('int')
        y = y.astype('int')
        img = np.full((H, W, 3), fill_value=255, dtype='uint8')
        mask = np.zeros((H, W), dtype='int32')
        p = p.astype('int')
        # p[p == 0] = -1
        mask1 = (x >= 0) & (y >= 0) & (W > x) & (H > y)
        mask[y[mask1], x[mask1]] = p
        img[mask == 0] = [255, 255, 255]
        img[mask == -1] = [255, 0, 0]
        img[mask == 1] = [0, 0, 255]

        self.plot_data(img, path, is_save)

        return img

    def plot_event_3d(self, fig, inp_event_list, inp_resolution, gt_event_list=None, gt_resolution=None):
        """
        event_list: np.ndarray, Nx4, [x, y, t, p], p:[-1, 1]
        """
        inp_x, inp_y, inp_t, inp_p = inp_event_list[:, 0], inp_event_list[:, 1], inp_event_list[:, 2], inp_event_list[:, 3]
        inp_y = inp_resolution[0] - inp_y
        if gt_event_list is not None:
            gt_x, gt_y, gt_t, gt_p = gt_event_list[:, 0], gt_event_list[:, 1], gt_event_list[:, 2], gt_event_list[:, 3]
            gt_y = gt_resolution[0] - gt_y

        # fig = plt.figure()
        if gt_event_list is not None:
            inp_ax = fig.add_axes([-0.1, 0.25, 0.7, 0.7], projection='3d')
            gt_ax = fig.add_axes([0.4, 0.25, 0.7, 0.7], projection='3d')
        else:
            inp_ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        inp_ax.scatter(inp_x[inp_p == 1], inp_t[inp_p == 1], inp_y[inp_p == 1], c='b', marker='.', s=1)
        inp_ax.scatter(inp_x[inp_p == -1], inp_t[inp_p == -1], inp_y[inp_p == -1], c='r', marker='.', s=1)
        inp_ax.set_xlabel('x');inp_ax.set_ylabel('t');inp_ax.set_zlabel('y')

        if gt_event_list is not None:
            gt_ax.scatter(gt_x[gt_p == 1], gt_t[gt_p == 1], gt_y[gt_p == 1], c='b', marker='.', s=1)
            gt_ax.scatter(gt_x[gt_p == -1], gt_t[gt_p == -1], gt_y[gt_p == -1], c='r', marker='.', s=1)
            gt_ax.set_xlabel('x');gt_ax.set_ylabel('t');gt_ax.set_zlabel('y')

        # plt.show()
    
    def plot_scale_inp_event(self, inputs, path, scale_resolution):
        def get_data(inputs):
            inp_coords = inputs['inp_ori_list_sparse']['batch_coords'][:, 1:4]
            inp_p = inputs['inp_ori_list_sparse']['batch_feats'][:, 3:4]
            inp = torch.cat([inp_coords, inp_p], dim=1)
            return inp

        scale_inp_event_list = get_data(inputs)
        scale_inp_cnt = events_to_channels(scale_inp_event_list[:, 0], scale_inp_event_list[:, 1], scale_inp_event_list[:, 3], scale_resolution)
        self.plot_event_cnt(scale_inp_cnt.numpy().transpose(1, 2, 0),
                                 path=path,
                                 color_scheme="green_red",
                                 is_save=True,
                                 is_black_background=False)


def main():
    # dataloader config
    dataloader_config = {
            'time_resolution': 256,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': False,
            'dataset': {
                'scale': 2,
                'ori_scale': 'down2',
                'time_bins': 2,
                'need_gt_frame': True,
                'mode': 'events', # events/time/frames
                'window': 5000,
                'sliding_window': 0,
                'data_augment': {
                    'enabled': False,
                    'augment': ["Horizontal", "Vertical", "Polarity"],
                    'augment_prob': [0.5, 0.5, 0.5],
                },
                'hot_filter': {
                    'enabled': False,
                    'max_px': 100,
                    'min_obvs': 5,
                    'max_rate': 0.8,
                }
            }
        }

    event_vis = event_visualisation()

    path_to_file = ''
    dataname = os.path.basename(path_to_file)
    save_path = f'/disk/work/output/chek_dataset/{dataname}'
    is_save = False

    inp_stack_path = os.path.join(save_path, 'inp_stack')
    gt_stack_path = os.path.join(save_path, 'gt_stack')
    gt_frame_path = os.path.join(save_path, 'gt_frame')
    inp_cnt_path = os.path.join(save_path, 'inp_cnt')
    gt_cnt_path = os.path.join(save_path, 'gt_cnt')
    inp_event_img_path = os.path.join(save_path, 'inp_event_img')
    gt_event_img_path = os.path.join(save_path, 'gt_event_img')
    inp_scale_cnt_path = os.path.join(save_path, 'inp_scale_cnt')

    if is_save:
        os.makedirs(inp_stack_path, exist_ok=False)
        os.makedirs(gt_stack_path, exist_ok=False)
        os.makedirs(gt_frame_path, exist_ok=False)
        os.makedirs(inp_cnt_path, exist_ok=False)
        os.makedirs(gt_cnt_path, exist_ok=False)
        os.makedirs(inp_event_img_path, exist_ok=False)
        os.makedirs(gt_event_img_path, exist_ok=False)
        os.makedirs(inp_scale_cnt_path, exist_ok=False)

    dataloader = InferenceHDF5DataLoader(path_to_file, dataloader_config)
    gt_sensor_resolution = dataloader.dataset.gt_sensor_resolution
    inp_sensor_resolution = dataloader.dataset.inp_sensor_resolution

    for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        # plot input
        event_vis.plot_event_cnt(inputs['inp_cnt'][0].numpy().transpose(1, 2, 0),
                                 path=os.path.join(inp_cnt_path, f'{idx:09d}.png'),
                                 color_scheme="green_red",
                                 is_save=is_save,
                                 is_black_background=False)
        event_vis.plot_event_img(inputs['inp_ori_list'][0].numpy(),
                                 resolution=inp_sensor_resolution,
                                 path=os.path.join(inp_event_img_path, f'{idx:09d}.png'),
                                 is_save=is_save)
        event_vis.plot_event_stack(inputs['inp_stack'][0].numpy().transpose(1, 2, 0),
                                   path=os.path.join(inp_stack_path, f'{idx:09d}.png'),
                                   is_save=is_save)
        # plot GT
        event_vis.plot_event_cnt(inputs['gt_cnt'][0].numpy().transpose(1, 2, 0),
                                 path=os.path.join(gt_cnt_path, f'{idx:09d}.png'),
                                 color_scheme="green_red",
                                 is_save=is_save,
                                 is_black_background=False)
        event_vis.plot_event_img(inputs['gt_list'][0].numpy(),
                                 resolution=gt_sensor_resolution,
                                 path=os.path.join(gt_event_img_path, f'{idx:09d}.png'),
                                 is_save=is_save)
        event_vis.plot_event_stack(inputs['gt_stack'][0].numpy().transpose(1, 2, 0),
                                   path=os.path.join(gt_stack_path, f'{idx:09d}.png'),
                                   is_save=is_save)
        event_vis.plot_frame((inputs['gt_img'][0].squeeze(0).numpy() * 255).astype('uint8'),
                                   path=os.path.join(gt_frame_path, f'{idx:09d}.png'),
                                   is_save=is_save)

        # plot event 3D
        # event_vis.plot_event_3d(inputs['inp_ori_list'][0].numpy(), inp_sensor_resolution,
        #                         inputs['gt_list'][0].numpy(), gt_sensor_resolution)

        # plot scaled input events
        # event_vis.plot_scale_inp_event(inputs=inputs, path=os.path.join(inp_scale_cnt_path, f'{idx:09d}.png'), scale_resolution=gt_sensor_resolution)

        plt.show()
        plt.close('all')


pause = False
def plot_event_3d_ori():
    dataloader_config = {
        'time_resolution': 256,
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': False,
        'dataset': {
            'scale': 2,
            'ori_scale': 'down2',
            'time_bins': 16,
            'need_gt_frame': True,
            'mode': 'events', # events/time/frames
            'window': 4000,
            'sliding_window': 3900,
            'data_augment': {
                'enabled': False,
                'augment': ["Horizontal", "Vertical", "Polarity"],
                'augment_prob': [0.5, 0.5, 0.5],
            },
            'hot_filter': {
                'enabled': False,
                'max_px': 100,
                'min_obvs': 5,
                'max_rate': 0.8,
            }
        }
    }

    event_vis = event_visualisation()

    path_to_file = ''
    dataloader = InferenceHDF5DataLoader(path_to_file, dataloader_config)
    gt_sensor_resolution = dataloader.dataset.gt_sensor_resolution
    inp_sensor_resolution = dataloader.dataset.inp_sensor_resolution

    def onClick(event):
        # plt.pause(0.5)
        global pause
        pause ^= True
    
    def on_key_press(event):
        # if event.key == 'a':
        print(event.key)
            # plt.pause(0.5)
            # fig.canvas.draw_idle()

    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    global pause
    for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        # event_vis.plot_event_3d(inputs['inp_ori_list'][0].numpy(), inp_sensor_resolution,
        #                         inputs['gt_list'][0].numpy(), gt_sensor_resolution)
        inp_coords = inputs['inp_ori_list_sparse']['batch_coords'][:, 1:4]
        inp_p = inputs['inp_ori_list_sparse']['batch_feats'][:, 3:4]
        inp = torch.cat([inp_coords, inp_p], dim=1)
        gt_coords = inputs['gt_list_sparse']['batch_coords'][:, 1:4]
        gt_p = inputs['gt_list_sparse']['batch_feats'][:, 3:4]
        gt = torch.cat([gt_coords, gt_p], dim=1)

        event_vis.plot_event_3d(fig,
                                inp.numpy(), gt_sensor_resolution,
                                gt.numpy(), gt_sensor_resolution)
        plt.pause(0.1)
        # if pause:
        #     plt.ioff()
        #     plt.show()
        #     pause = False
        #     plt.ion()
        fig.clf()
        # plt.close('all')

    plt.ioff()


def plot_event_3d():
    def onClick(event):
        global pause
        pause ^= True

    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onClick)

    def DataGenerator():
        def get_data(inputs):
            inp_coords = inputs['inp_ori_list_sparse']['batch_coords'][:, 1:4]
            inp_p = inputs['inp_ori_list_sparse']['batch_feats'][:, 3:4]
            inp = torch.cat([inp_coords, inp_p], dim=1)
            gt_coords = inputs['gt_list_sparse']['batch_coords'][:, 1:4]
            gt_p = inputs['gt_list_sparse']['batch_feats'][:, 3:4]
            gt = torch.cat([gt_coords, gt_p], dim=1)
            return inp, gt

        dataloader_config = {
            'time_resolution': 256,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
            'drop_last': False,
            'dataset': {
                'scale': 2,
                'ori_scale': 'down2',
                'time_bins': 16,
                'need_gt_frame': True,
                'mode': 'events', # events/time/frames
                'window': 2000,
                'sliding_window': 0,
                'data_augment': {
                    'enabled': False,
                    'augment': ["Horizontal", "Vertical", "Polarity"],
                    'augment_prob': [0.5, 0.5, 0.5],
                },
                'hot_filter': {
                    'enabled': False,
                    'max_px': 100,
                    'min_obvs': 5,
                    'max_rate': 0.8,
                }
            }
        }

        path_to_file = ''
        dataloader = InferenceHDF5DataLoader(path_to_file, dataloader_config)
        gt_sensor_resolution = dataloader.dataset.gt_sensor_resolution
        inp_sensor_resolution = dataloader.dataset.inp_sensor_resolution
        # data_iter = iter(dataloader)
        # data_len = len(dataloader)

        # data = get_data(data_iter.next())
        # idx = 0

        # while idx < data_len:
        #     if not pause:
        #         data = get_data(data_iter.next())
        #         idx += 1
        #     yield data[0].numpy(), gt_sensor_resolution, data[1].numpy(), gt_sensor_resolution


        for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)): 
            data = get_data(inputs)
            if not pause:
                yield data[0].numpy(), gt_sensor_resolution, data[1].numpy(), gt_sensor_resolution
            else:
                yield data[0].numpy(), gt_sensor_resolution, data[1].numpy(), gt_sensor_resolution

    def DataPlot(data):
        """
        event_list: np.ndarray, Nx4, [x, y, t, p], p:[-1, 1]
        """
        inp_event_list, inp_resolution, gt_event_list, gt_resolution = data[0], data[1], data[2], data[3]
        inp_x, inp_y, inp_t, inp_p = inp_event_list[:, 0], inp_event_list[:, 1], inp_event_list[:, 2], inp_event_list[:, 3]
        inp_y = inp_resolution[0] - inp_y
        if gt_event_list is not None:
            gt_x, gt_y, gt_t, gt_p = gt_event_list[:, 0], gt_event_list[:, 1], gt_event_list[:, 2], gt_event_list[:, 3]
            gt_y = gt_resolution[0] - gt_y

        # fig = plt.figure()
        if gt_event_list is not None:
            inp_ax = fig.add_axes([-0.1, 0.25, 0.7, 0.7], projection='3d')
            gt_ax = fig.add_axes([0.4, 0.25, 0.7, 0.7], projection='3d')
        else:
            inp_ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        inp_ax.scatter(inp_x[inp_p == 1], inp_t[inp_p == 1], inp_y[inp_p == 1], c='b', marker='.', s=1)
        inp_ax.scatter(inp_x[inp_p == -1], inp_t[inp_p == -1], inp_y[inp_p == -1], c='r', marker='.', s=1)
        inp_ax.set_xlabel('x');inp_ax.set_ylabel('t');inp_ax.set_zlabel('y')

        if gt_event_list is not None:
            gt_ax.scatter(gt_x[gt_p == 1], gt_t[gt_p == 1], gt_y[gt_p == 1], c='b', marker='.', s=1)
            gt_ax.scatter(gt_x[gt_p == -1], gt_t[gt_p == -1], gt_y[gt_p == -1], c='r', marker='.', s=1)
            gt_ax.set_xlabel('x');gt_ax.set_ylabel('t');gt_ax.set_zlabel('y')

    ani = animation.FuncAnimation(fig, DataPlot, DataGenerator, blit=False, interval=0.1, repeat=True)
    plt.show()


class PlotEvent3DFunc():
    def __init__(self):
        self.pause = False
        self.fig = plt.figure()
        self.inp_ax = self.fig.add_axes([-0.1, 0.25, 0.7, 0.7], projection='3d')
        self.inp_ax.set_xlabel('x');self.inp_ax.set_ylabel('t');self.inp_ax.set_zlabel('y')
        self.gt_ax = self.fig.add_axes([0.4, 0.25, 0.7, 0.7], projection='3d')
        self.gt_ax.set_xlabel('x');self.gt_ax.set_ylabel('t');self.gt_ax.set_zlabel('y')

        self.ani = animation.FuncAnimation(self.fig, self.DataPlot, self.DataGenerator, blit=False, interval=5, repeat=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    def DataPlot(self, data):
        inp_event_list, inp_resolution, gt_event_list, gt_resolution = data[0], data[1], data[2], data[3]
        inp_x, inp_y, inp_t, inp_p = inp_event_list[:, 0], inp_event_list[:, 1], inp_event_list[:, 2], inp_event_list[:, 3]
        inp_y = inp_resolution[0] - inp_y
        gt_x, gt_y, gt_t, gt_p = gt_event_list[:, 0], gt_event_list[:, 1], gt_event_list[:, 2], gt_event_list[:, 3]
        gt_y = gt_resolution[0] - gt_y

        self.inp_ax.scatter(inp_x[inp_p == 1], inp_t[inp_p == 1], inp_y[inp_p == 1], c='b', marker='.', s=1)
        self.inp_ax.scatter(inp_x[inp_p == -1], inp_t[inp_p == -1], inp_y[inp_p == -1], c='r', marker='.', s=1)
        self.gt_ax.scatter(gt_x[gt_p == 1], gt_t[gt_p == 1], gt_y[gt_p == 1], c='b', marker='.', s=1)
        self.gt_ax.scatter(gt_x[gt_p == -1], gt_t[gt_p == -1], gt_y[gt_p == -1], c='r', marker='.', s=1)

        plt.show()
        plt.pause(0.1)
        self.fig.clf()

    @staticmethod
    def DataGenerator():
        def get_data(inputs):
            inp_coords = inputs['inp_ori_list_sparse']['batch_coords'][:, 1:4]
            inp_p = inputs['inp_ori_list_sparse']['batch_feats'][:, 3:4]
            inp = torch.cat([inp_coords, inp_p], dim=1)
            gt_coords = inputs['gt_list_sparse']['batch_coords'][:, 1:4]
            gt_p = inputs['gt_list_sparse']['batch_feats'][:, 3:4]
            gt = torch.cat([gt_coords, gt_p], dim=1)
            return inp, gt

        dataloader_config = {
            'time_resolution': 256,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
            'drop_last': False,
            'dataset': {
                'scale': 2,
                'ori_scale': 'down2',
                'time_bins': 16,
                'need_gt_frame': True,
                'mode': 'events', # events/time/frames
                'window': 2000,
                'sliding_window': 0,
                'data_augment': {
                    'enabled': False,
                    'augment': ["Horizontal", "Vertical", "Polarity"],
                    'augment_prob': [0.5, 0.5, 0.5],
                },
                'hot_filter': {
                    'enabled': False,
                    'max_px': 100,
                    'min_obvs': 5,
                    'max_rate': 0.8,
                }
            }
        }

        path_to_file = ''
        dataloader = InferenceHDF5DataLoader(path_to_file, dataloader_config)
        gt_sensor_resolution = dataloader.dataset.gt_sensor_resolution
        inp_sensor_resolution = dataloader.dataset.inp_sensor_resolution

        for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)): 
            data = get_data(inputs)
            yield data[0].numpy(), gt_sensor_resolution, data[1].numpy(), gt_sensor_resolution

    def on_key_press(self, event):
        if event.key == 'r':
            print('Resume')
            self.ani.event_source.start()
        elif event.key == 'p':
            print('Pause')
            self.ani.event_source.stop()


class PlotEvent3D():
    def __init__(self, data_len=None):
        self.pause = False
        self.data_len = data_len
        self.movie = []
        self.fig = plt.figure()

        self.inp_ax = self.fig.add_axes([-0.1, 0.35, 0.7, 0.7], projection='3d')
        self.inp_ax.set_xlabel('x');self.inp_ax.set_ylabel('t');self.inp_ax.set_zlabel('y')

        self.gt_ax = self.fig.add_axes([0.4, 0.35, 0.7, 0.7], projection='3d')
        self.gt_ax.set_xlabel('x');self.gt_ax.set_ylabel('t');self.gt_ax.set_zlabel('y')

        self.frame_ax = self.fig.add_axes([0.35, 0., 0.35, 0.35])
        self.frame_ax.axis('off') # remove white border
        self.frame_ax.set_xticks([]);self.frame_ax.set_yticks([])

        self.CreateMovie()
        self.ani = animation.ArtistAnimation(self.fig, self.movie, interval=20, repeat=True)
        # self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()
        # self.ani.save('/disk/work/output/chek_dataset/basketball_1.gif', writer='pillow')

    def CreateMovie(self):
        def get_data(inputs):
            inp_coords = inputs['inp_ori_list_sparse']['batch_coords'][:, 1:4]
            inp_p = inputs['inp_ori_list_sparse']['batch_feats'][:, 3:4]
            inp = torch.cat([inp_coords, inp_p], dim=1)
            gt_coords = inputs['gt_list_sparse']['batch_coords'][:, 1:4]
            gt_p = inputs['gt_list_sparse']['batch_feats'][:, 3:4]
            gt = torch.cat([gt_coords, gt_p], dim=1)
            frame = (inputs['gt_img'][0].squeeze(0).numpy() * 255).astype('uint8')

            return inp.numpy(), gt_sensor_resolution, gt.numpy(), gt_sensor_resolution, frame

        dataloader_config = {
            'time_resolution': 256,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
            'drop_last': False,
            'dataset': {
                'scale': 2,
                'ori_scale': 'down2',
                'time_bins': 16,
                'need_gt_frame': True,
                'mode': 'events', # events/time/frames
                'window': 2000,
                'sliding_window': 0,
                'data_augment': {
                    'enabled': False,
                    'augment': ["Horizontal", "Vertical", "Polarity"],
                    'augment_prob': [0.5, 0.5, 0.5],
                },
                'hot_filter': {
                    'enabled': False,
                    'max_px': 100,
                    'min_obvs': 5,
                    'max_rate': 0.8,
                }
            }
        }

        path_to_file = ''
        dataloader = InferenceHDF5DataLoader(path_to_file, dataloader_config)
        gt_sensor_resolution = dataloader.dataset.gt_sensor_resolution
        inp_sensor_resolution = dataloader.dataset.inp_sensor_resolution

        for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)): 
            movie_split = []
            data = get_data(inputs)
            inp_event_list, inp_resolution, gt_event_list, gt_resolution, frame = data[0], data[1], data[2], data[3], data[4]
            inp_x, inp_y, inp_t, inp_p = inp_event_list[:, 0], inp_event_list[:, 1], inp_event_list[:, 2], inp_event_list[:, 3]
            inp_y = inp_resolution[0] - inp_y
            gt_x, gt_y, gt_t, gt_p = gt_event_list[:, 0], gt_event_list[:, 1], gt_event_list[:, 2], gt_event_list[:, 3]
            gt_y = gt_resolution[0] - gt_y

            movie_split.append(self.inp_ax.scatter(inp_x[inp_p == 1], inp_t[inp_p == 1], inp_y[inp_p == 1], c='b', marker='.', s=1))
            movie_split.append(self.inp_ax.scatter(inp_x[inp_p == -1], inp_t[inp_p == -1], inp_y[inp_p == -1], c='r', marker='.', s=1))
            movie_split.append(self.gt_ax.scatter(gt_x[gt_p == 1], gt_t[gt_p == 1], gt_y[gt_p == 1], c='b', marker='.', s=1))
            movie_split.append(self.gt_ax.scatter(gt_x[gt_p == -1], gt_t[gt_p == -1], gt_y[gt_p == -1], c='r', marker='.', s=1))
            movie_split.append(self.frame_ax.imshow(frame, cmap='gray'))
            self.movie.append(movie_split)

            if self.data_len is not None:
                if idx >= self.data_len:
                    break

    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()
        self.pause ^= True

    def set_view(self, elev, azim):
        self.inp_ax.view_init(elev, azim) 
        self.gt_ax.view_init(elev, azim) 

    def on_key_press(self, event):
        if event.key == 'r':
            print('Resume')
            self.ani.event_source.start()
        elif event.key == 'p':
            print('Pause')
            self.ani.event_source.stop()
        elif event.key == 'v':
            print('print view')
            print(f'elev: {self.inp_ax.elev}')
            print(f'azim: {self.inp_ax.azim}')
        elif event.key == '1':
            print('set view 1')
            self.set_view(elev=0, azim=-90)
            print(f'elev: {self.inp_ax.elev}')
            print(f'azim: {self.inp_ax.azim}')
        elif event.key == '2':
            print('set view 2')
            self.set_view(elev=30, azim=-60)
            print(f'elev: {self.inp_ax.elev}')
            print(f'azim: {self.inp_ax.azim}')
        elif event.key == '3':
            print('set view 3')
            self.set_view(elev=30, azim=-120)
            print(f'elev: {self.inp_ax.elev}')
            print(f'azim: {self.inp_ax.azim}')
        elif event.key == '4':
            print('set view 4')
            self.set_view(elev=-30, azim=-60)
            print(f'elev: {self.inp_ax.elev}')
            print(f'azim: {self.inp_ax.azim}')
        elif event.key == '5':
            print('set view 5')
            self.set_view(elev=-30, azim=-120)
            print(f'elev: {self.inp_ax.elev}')
            print(f'azim: {self.inp_ax.azim}')



if __name__ == '__main__':
    main()
    # eventplot = PlotEvent3D(200)
