import os
from numpy import mean
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import time
import torch.nn.functional as f
from myutils.utils import *
from dataloader.h5dataset import *
from dataloader.h5dataloader import *
from loss import *
from dataloader.encodings import *
from myutils.vis_events.matplotlib_plot_events import *
from models.BMCNet import BMCNet


def infer_body(dataloader_config, data_path, esr_model,
               event_img_path, logger: Logger_yaml,
               device, vis: event_visualisation, metrics):
    mse = metrics['mse']
    scale = dataloader_config['dataset']['scale']
    n_c = 128

    # build dataset
    logger.log_dict(dataloader_config, 'eval_datasetloader_config')
    dataloader = InferenceHDF5DataLoaderSequence(data_path, dataloader_config)
    gt_sensor_resolution = dataloader.dataset.gt_sensor_resolution
    inp_sensor_resolution = dataloader.dataset.inp_sensor_resolution

    metric_track = MetricTracker(['esr_mse', 'bicubic_mse', 'time', 'params', 'macs'])

    # prepare img paths
    os.makedirs(os.path.join(event_img_path, 'lr_event_img'), exist_ok=False)
    os.makedirs(os.path.join(event_img_path, 'hr_esr_event_img'), exist_ok=False)
    os.makedirs(os.path.join(event_img_path, 'hr_bicubic_event_img'), exist_ok=False)
    os.makedirs(os.path.join(event_img_path, 'hr_gt_event_img'), exist_ok=False)

    mid_idx = 1
    mid_grp = (dataloader_config['dataset']['sequence']['sequence_length'] - dataloader_config['dataset']['sequence']['seqn'] + 1) // 2
    metric_track.reset()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    init = True
    for i, inputs_seq in enumerate(tqdm(dataloader, total=len(dataloader))):

        inputs = inputs_seq[0]

        input_stack = inputs['inp_cnt'].transpose(1, 2) 
        inp_cnt = inputs['inp_cnt'][:, mid_idx]
        gt_stack = inputs['gt_cnt'][:, 1]  # Bx2xkHxkW

        starter.record()
        if init:
            init_temp = torch.zeros_like(input_stack[:, 0:1, 0, :, :])
            init_o = init_temp.repeat(1, scale * scale * 2, 1, 1).to(device)
            init_h = init_temp.repeat(1, n_c, 1, 1).to(device)
            init_h_p = init_temp.repeat(1, n_c, 1, 1).to(device)
            init_h_n = init_temp.repeat(1, n_c, 1, 1).to(device)
            h, hp, hn, prediction = esr_model(input_stack.to(device), init_h, init_h_p, init_h_n, init_o, init)
            init = False
        else:
            h, hp, hn, prediction = esr_model(input_stack.to(device), h, hp, hn, prediction, init)

        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)

        if i == 0:
            params = sum(p.numel() for p in esr_model.parameters())
            metric_track.update('params', params / 1e6)

        # forward
        esr_cnt = prediction
        gt_cnt = gt_stack
        esr_cnt = esr_cnt.cpu()
        if esr_cnt.size()[-2:] != gt_cnt.size()[-2:]:
            esr_cnt = f.interpolate(esr_cnt, size=gt_cnt.size()[-2:], mode='bicubic', align_corners=False)
        bicubic_cnt = f.interpolate(inp_cnt, size=gt_sensor_resolution, mode='bicubic', align_corners=False)

        # metrics computation and save
        esr_mse = mse(esr_cnt, gt_cnt)
        bicubic_mse = mse(bicubic_cnt, gt_cnt)
        metric_track.update('esr_mse', esr_mse.item())
        metric_track.update('bicubic_mse', bicubic_mse.item())
        metric_track.update('time', total_time)

        # images save
        vis.plot_event_cnt(inputs['inp_cnt'][0, mid_idx].cpu().numpy().transpose(1, 2, 0), is_save=True,
                           path=os.path.join(event_img_path, 'lr_event_img', '{:09d}.png'.format(i)))
        vis.plot_event_cnt(bicubic_cnt[0].cpu().numpy().transpose(1, 2, 0), is_save=True,
                           path=os.path.join(event_img_path, 'hr_bicubic_event_img', '{:09d}.png'.format(i)))
        vis.plot_event_cnt(esr_cnt[0].cpu().round().numpy().transpose(1, 2, 0), is_save=True,
                           path=os.path.join(event_img_path, 'hr_esr_event_img', '{:09d}.png'.format(i)))
        vis.plot_event_cnt(inputs['gt_cnt'][0, mid_idx].cpu().numpy().transpose(1, 2, 0), is_save=True,
                           path=os.path.join(event_img_path, 'hr_gt_event_img', '{:09d}.png'.format(i)))
        plt.close('all')

    result = metric_track.result()
    logger.log_dict(result, 'evaluation results')

    return result


def load_model(dataloader_config, model_path, device, n_c = 128, n_b = 5):
    esr_cpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(f'Load esr model from: {model_path}...')

    # build ESR
    esr_model = BMCNet(dataloader_config['dataset']['scale'], n_c, n_b)
    esr_model.load_state_dict(esr_cpt)
    esr_model.to(device)
    esr_model.eval()

    return esr_model


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_list', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--data_list', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--infer_mode', type=int, required=True, choices=[0, 1, 2])
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--scale', type=int, default=None)
    parser.add_argument('--seqn', type=int, default=None)
    parser.add_argument('--seql', type=int, default=None)
    parser.add_argument('--step_size', type=int, default=None)
    parser.add_argument('--time_bins', type=int, default=None)
    parser.add_argument('--ori_scale', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--sliding_window', type=int, default=None)
    parser.add_argument('--need_gt_frame', default=False, action='store_true')
    parser.add_argument('--need_gt_events', default=False, action='store_true')

    return parser.parse_args()


@torch.no_grad()
def main():
    SCALE = 4
    SEQN = 3
    SEQL = 9
    STEP_SIZE = None
    TIME_BINS = 1
    ORI_SCALE = 'down4'
    MODE = 'events'  # events/time/frame
    WINDOW = 2048
    SLIDING_WINDOW = 1024
    NEED_GT_FRAME = True
    NEED_GT_EVENTS = True

    dataloader_config = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': False,
        'dataset': {
            'scale': SCALE,
            'ori_scale': ORI_SCALE,
            'time_bins': TIME_BINS,
            'need_gt_frame': NEED_GT_FRAME,
            'need_gt_events': NEED_GT_EVENTS,
            'mode': MODE,  # events/time/frames
            'window': WINDOW,
            'sliding_window': SLIDING_WINDOW,
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
            },
            'sequence': {
                'sequence_length': SEQL,
                'seqn': SEQN,
                'step_size': STEP_SIZE,
                'pause': {
                    'enabled': False,
                    'proba_pause_when_running': 0.05,
                    'proba_pause_when_paused': 0.9,
                },
            },
        },
    }

    flags = get_flags()

    infer_mode = flags.infer_mode
    model_path = flags.model_path
    model_list = flags.model_list
    data_path = flags.data_path
    data_list = flags.data_list
    device = torch.device(flags.device)
    output_path = flags.output_path
    os.makedirs(output_path, exist_ok=True)

    scale = flags.scale
    seqn = flags.seqn
    seql = flags.seql
    step_size = flags.step_size
    time_bins = flags.time_bins
    ori_scale = flags.ori_scale
    mode = flags.mode
    window = flags.window
    sliding_window = flags.sliding_window
    need_gt_frame = flags.need_gt_frame
    need_gt_events = flags.need_gt_events

    if scale is not None:
        dataloader_config['dataset'].update({'scale': scale})
    if seqn is not None:
        dataloader_config['dataset']['sequence'].update({'seqn': seqn})
    if seql is not None:
        dataloader_config['dataset']['sequence'].update({'sequence_length': seql})
    if step_size is not None:
        dataloader_config['dataset']['sequence'].update({'step_size': step_size})
    if time_bins is not None:
        dataloader_config['dataset'].update({'time_bins': time_bins})
    if ori_scale is not None:
        dataloader_config['dataset'].update({'ori_scale': ori_scale})
    if mode is not None:
        dataloader_config['dataset'].update({'mode': mode})
    if window is not None:
        dataloader_config['dataset'].update({'window': window})
    if sliding_window is not None:
        dataloader_config['dataset'].update({'sliding_window': sliding_window})
    if need_gt_frame is not None:
        dataloader_config['dataset'].update({'need_gt_frame': need_gt_frame})
    if need_gt_events is not None:
        dataloader_config['dataset'].update({'need_gt_events': need_gt_events})

    vis = event_visualisation()
    metrics = {
        'mse': nn.MSELoss(),
    }

    if infer_mode == 1:
        assert os.path.isfile(model_path)
        assert os.path.isfile(data_list)

        esr_model = load_model(dataloader_config, model_path, device)

        data_list = pd.read_csv(data_list, header=None).values.flatten().tolist()

        logger_all = Logger_yaml(os.path.join(output_path, 'inference_all.yml'))
        logger_all.log_info(f'inference {model_path} on {data_list}')

        results = []
        for data_path in tqdm(data_list):
            print(f'processing {data_path}')
            data_name = os.path.basename(data_path)
            root_path = os.path.join(output_path, data_name)
            event_img_path = os.path.join(root_path, 'event_img')
            os.makedirs(root_path, exist_ok=False)
            os.makedirs(event_img_path, exist_ok=False)

            logger = Logger_yaml(os.path.join(root_path, 'inference.yml'))
            logger.log_info(f'inference {model_path} on {data_path}')
            args = {
                'dataloader_config': dataloader_config,
                'data_path': data_path,
                'esr_model': esr_model,
                'event_img_path': event_img_path,
                'logger': logger,
                'device': device,
                'vis': vis,
                'metrics': metrics,
            }
            result = infer_body(**args)
            result['data_name'] = data_name
            results.append(result)

        results_dict = defaultdict(dict)
        results_mean = defaultdict(list)
        for entry in results:
            data_name = entry.pop('data_name')
            for k, v in entry.items():
                results_dict[k][data_name] = v
                results_mean[k].append(v)
        for k, v in results_mean.items():
            results_mean[k] = float(mean(v))

        logger_all.log_dict(dict(results_dict), 'breakdown results for each data')
        logger_all.log_dict(dict(results_mean), 'mean results for the whole data')

    else:
        raise Exception(f'Not support infer mode {infer_mode}')


if __name__ == '__main__':
    main()