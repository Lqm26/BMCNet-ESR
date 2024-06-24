import torch
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.optim import *
import torch.nn.functional as f
from torch.nn.parallel import DistributedDataParallel as ddp
import collections
from torch.optim.lr_scheduler import *
from numpy import inf
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
# local modules
from config.parser import YAMLParser
from dataloader.h5dataloader import HDF5DataLoader, HDF5DataLoaderSequence
from loss import *
from myutils.utils import *
from logger import *
from myutils.timers import Timer
from myutils.vis_events.visualization import *
from myutils.vis_events.matplotlib_plot_events import *
from dataloader.encodings import *
import time
from models.BMCNet import BMCNet

def init_seeds(seed=0, cuda_deterministic=True):
    print(f'seed:{seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.backends.cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        raise Exception('Only support DDP')

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    dist_url = 'env://'
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    dist.barrier()
    setup_for_distributed(rank == 0)

    return gpu


class Trainer:
    def __init__(self, args):
        # config parser
        self.config_parser = args['config_parser']
        # dataloader
        self.train_dataloader = args['train_dataloader']
        self.valid_dataloader = args['valid_dataloader']
        # models
        self.esr_model = args['esr_model']
        # loss fts
        self.esr_loss = args['esr_loss']
        # optimizers
        self.esr_optimizer = args['esr_optimizer']
        # lr scheduler
        self.esr_lr_scheduler = args['esr_lr_scheduler']
        # metadata
        self.logger = args['logger']
        self.device = args['device']

        self.monitor = self.config_parser['trainer'].get('monitor', 'off')
        self.checkpoint_dir = self.config_parser.save_dir
        self.inp_sensor_resolution = self.train_dataloader.dataset.datasets[0].inp_sensor_resolution
        self.gt_sensor_resolution = self.train_dataloader.dataset.datasets[0].gt_sensor_resolution
        self.do_validation = self.valid_dataloader is not None
        self.scale = args['scale']
        self.n_c = args['n_c']

        # training mode setting
        is_epoch_based_train = self.config_parser['trainer']['epoch_based_train']['enabled']
        is_iteration_based_train = self.config_parser['trainer']['iteration_based_train']['enabled']
        if (is_epoch_based_train and is_iteration_based_train) or \
                (not is_epoch_based_train and not is_iteration_based_train):
            raise Exception('Please set correct training mode in the configuration file!')
        elif is_epoch_based_train:
            # metadata for epoch-based training
            if dist.get_rank() == 0:
                self.logger.info('Apply epoch-based training...')
            self.training_mode = 'epoch_based_train'
            self.epochs = self.config_parser['trainer']['epoch_based_train']['epochs']
            self.start_epoch = 1
            self.len_epoch = len(self.train_dataloader)
            self.save_period = self.config_parser['trainer']['epoch_based_train']['save_period']
            self.train_log_step = max(len(self.train_dataloader) \
                                      // self.config_parser['trainer']['epoch_based_train']['train_log_step'], 1)
            self.valid_log_step = max(len(self.valid_dataloader) \
                                      // self.config_parser['trainer']['epoch_based_train']['valid_log_step'], 1)
            self.valid_step = self.config_parser['trainer']['epoch_based_train']['valid_step']
        elif is_iteration_based_train:

            self.logger.info('Apply iteration-based training...')
            self.training_mode = 'iteration_based_train'
            self.iterations = int(self.config_parser['trainer']['iteration_based_train']['iterations'])
            self.len_epoch = len(self.train_dataloader)
            self.save_period = self.config_parser['trainer']['iteration_based_train']['save_period']
            self.train_log_step = self.config_parser['trainer']['iteration_based_train']['train_log_step']
            self.valid_log_step = self.config_parser['trainer']['iteration_based_train']['valid_log_step']
            self.valid_step = self.config_parser['trainer']['iteration_based_train']['valid_step']
            self.lr_change_rate = self.config_parser['trainer']['iteration_based_train']['lr_change_rate']

        # visualization tool
        self.vis = event_visualisation()

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.config_parser['trainer'].get('early_stop', inf)

        # setup visualization writer instance
        self.writer = TensorboardWriter(self.config_parser.log_dir, self.logger,
                                            self.config_parser['trainer']['tensorboard'])

        # setup metric tracker
        train_mt_keys = ['train_mse_loss', 'train_loss']
        valid_mt_keys = ['valid_mse_loss', 'valid_loss']
        self.train_metrics = MetricTracker(train_mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(valid_mt_keys, writer=self.writer)


    def train(self):
        """
        Full training logic
        """
        if self.training_mode == 'epoch_based_train':
            self.epoch_based_training()
        elif self.training_mode == 'iteration_based_train':
            self.iteration_based_training()
        else:
            raise Exception('Incorrect training config!')

    def iteration_based_training(self):
        """
        Iteration-based training logic
        """
        valid_stamp = 1
        epoch = 0
        stop_training = False
        complete_training = False
        lamda = 0.01
        self.mid_idx = (self.train_dataloader.seqn - 1) // 2
        self.not_improved_count = 0

        self.esr_model.train()
        self.train_metrics.reset()

        while True:
            if stop_training or complete_training:
                break
            if epoch == 0:
                num_params = sum(p.numel() for p in self.esr_model.parameters() if p.requires_grad)
                print(num_params)

            for idx, inputs_seq in enumerate(self.train_dataloader):
                iter_idx = idx + self.len_epoch * epoch
                best = False
                self.esr_optimizer.zero_grad()
                loss = 0
                init = True

                for inputs in inputs_seq:
                    # forward pass
                    input_stack = inputs['inp_cnt'].transpose(1, 2) 

                    gt_stack = inputs['gt_cnt'][:, 1].to(self.device)  # Bx2xkHxkW
                        
                    if init:
                        init_temp = torch.zeros_like(input_stack[:, 0:1, 0, :, :])
                        init_o = init_temp.repeat(1, self.scale * self.scale * 2, 1, 1).to(self.device)
                        init_h = init_temp.repeat(1, self.n_c, 1, 1).to(self.device)
                        init_h_p = init_temp.repeat(1, self.n_c, 1, 1).to(self.device)
                        init_h_n = init_temp.repeat(1, self.n_c, 1, 1).to(self.device)
                        h, hp, hn, prediction = self.esr_model(input_stack.to(self.device), init_h, init_h_p, init_h_n, init_o, init)
                        init = False
                    else:
                        h, hp, hn, prediction = self.esr_model(input_stack.to(self.device), h, hp, hn, prediction, init)

                    # loss and backward pass
                    if prediction.size()[-2:] != gt_stack.size()[-2:]:
                        scale_prediction = f.interpolate(prediction, size=gt_stack.size()[-2:], mode='bicubic',
                                                         align_corners=False)
                    else:
                        scale_prediction = prediction

                    mse_loss = self.esr_loss['mse'](scale_prediction, gt_stack)
                    loss += mse_loss

                loss.backward()
                self.esr_optimizer.step()

                # reduce losses over all GPUs for logging purposes
                reduced_mse_loss = reduce_tensor(mse_loss)
                reduced_loss = reduce_tensor(loss)

                # setup log info
                log_step = iter_idx
                learning_rate = self.esr_lr_scheduler.get_last_lr()[0]
                self.writer.set_step(iter_idx)
                self.train_metrics.update('train_mse_loss', reduced_mse_loss.item())
                self.train_metrics.update('train_loss', reduced_loss.item())
                self.writer.writer.add_scalar(f'learning rate', learning_rate, global_step=log_step)
                if iter_idx % self.train_log_step == 0:
                    msg = 'Train Epoch: {} {} Iteration: {} {}'.format(epoch + 1,
                                                                       self._progress(idx, self.train_dataloader,
                                                                                      is_train=False), \
                                                                       iter_idx, self._progress(iter_idx,
                                                                                                self.train_dataloader,
                                                                                                is_train=True))
                    msg += ' {}: {:.4e}'.format('train_mse_loss', reduced_mse_loss.item())
                    msg += ' {}: {:.4e}'.format('train_loss', reduced_loss.item())
                    msg += ' {}: {:.4e}'.format('learning rate', learning_rate)
                    self.logger.info(msg)

                    
                # do validation
                if self.do_validation:
                    if iter_idx % self.valid_step == 0 and iter_idx != 0:
                        with torch.no_grad():
                            val_log = self._valid(valid_stamp)
                            # plot stamp train & valid logs
                            for key, value in val_log.items():
                                self.writer.writer.add_scalar(f'stamp_{key}', value, global_step=valid_stamp)
                            self.writer.writer.add_scalar(f'stamp_train_mse_loss', reduced_mse_loss.item(),
                                                          global_step=valid_stamp)
                            self.writer.writer.add_scalar(f'stamp_train_loss', reduced_loss.item(),
                                                          global_step=valid_stamp)
                            log = {'Valid stamp': valid_stamp}
                            log.update(val_log)
                            for key, value in log.items():
                                print(value)
                                self.logger.info('    {:25s}: {}'.format(str(key), value))
                            # evaluate model performance
                            stop_training, best = self.eval_model_performance(val_log)
                            if stop_training:
                                break
                            valid_stamp += 1

                # save model
                if (iter_idx % self.save_period == 0 and iter_idx != 0) or best:
                        self._save_checkpoint(iter_idx, save_best=best)

                # change learning rate
                if self.esr_lr_scheduler is not None:
                    if iter_idx % self.lr_change_rate == 0 and iter_idx != 0 \
                            and self.esr_lr_scheduler.get_last_lr()[0] >= 1e-5:
                        self.esr_lr_scheduler.step()

                if iter_idx + 1 == self.iterations:
                    self.logger.info('Training completes!')
                    complete_training = True
                    break

            epoch += 1

    def epoch_based_training(self):
        """
        Epoch-based training logic
        """
        self.not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_dataloader.sampler.set_epoch(epoch)

            with Timer('Time of training one epoch', self.logger):
                epoch_result = self._train_epoch(epoch)

            # plot epoch average statics
            if dist.get_rank() == 0:
                for key, value in epoch_result.items():
                    self.writer.writer.add_scalar(f'epoch_{key}', value, global_step=epoch)
                # save log informations into log dict
                log = {'epoch': epoch}
                log.update(epoch_result)
                # print log informations to the screen
                for key, value in log.items():
                    self.logger.info('    {:25s}: {}'.format(str(key), value))

            # evaluate model performance
            stop_training, best = self.eval_model_performance(epoch_result)
            if stop_training:
                break

            # save model
            if dist.get_rank() == 0:
                if epoch % self.save_period == 0 or best:
                    self._save_checkpoint(epoch, save_best=best)

            # sync all processes
            dist.barrier()

        # complete training
        if dist.get_rank() == 0:
            self.logger.info('Training completes!')

    def eval_model_performance(self, log):
        """
        Evaluate model performance according to configured metric
        log: log includes validation metric
        """
        if self.monitor == 'off':
            self.logger.info('Please set the correct metric to evaluate model!')
        else:
            self.logger.info(
                f'Evaluate current model using metric "{self.mnt_metric}", and save the current best model...')

        best = False
        is_KeyError = False
        stop_training = False

        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                           (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                is_KeyError = False
            except KeyError:
                self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Ignore this stamp where using this metric to monitor.".format(self.mnt_metric))
                is_KeyError = True
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
                best = True
            elif not is_KeyError:
                self.not_improved_count += 1

            if self.not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} stamps. "
                                     "Training stops.".format(self.early_stop))
                stop_training = True

        return stop_training, best

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.esr_model.train()

        self.train_metrics.reset()
        for batch_idx, inputs in enumerate(self.train_dataloader):

            self.esr_optimizer.zero_grad()

            # lr forward pass
            inp_sparse = ME.SparseTensor(features=inputs['inp_ori_list_sparse']['batch_feats'],
                                         coordinates=inputs['inp_ori_list_sparse']['batch_coords'],
                                         tensor_stride=self.config_parser['INPUT_TENSOR_STRIDE'],
                                         device=self.device)
            gt_sparse = ME.SparseTensor(features=inputs['gt_list_sparse']['batch_feats'],
                                        coordinates=inputs['gt_list_sparse']['batch_coords'],
                                        tensor_stride=1,
                                        device=self.device)
            pred_logits, gt_logits, gt_pol, pred_sparse = self.esr_model(inp_sparse, gt_sparse)

            # loss and backward pass
            num_layers = len(pred_logits)
            bce_loss = 0
            for pred_logit, gt_logit in zip(pred_logits, gt_logits):
                curr_bce_loss = self.esr_loss['bce'](pred_logit.F.squeeze(), gt_logit.type(pred_logit.dtype))
                bce_loss += curr_bce_loss
            bce_loss = bce_loss / num_layers
            mse_loss = self.esr_loss['mse'](pred_sparse.F.squeeze(), gt_pol)
            loss = bce_loss + mse_loss
            loss.backward()
            self.esr_optimizer.step()

            # reduce losses over all GPUs for logging purposes
            reduced_bce_loss = reduce_tensor(bce_loss)
            reduced_mse_loss = reduce_tensor(mse_loss)
            reduced_loss = reduce_tensor(loss)

            # setup log info
            if dist.get_rank() == 0:
                log_step = (epoch - 1) * self.len_epoch + batch_idx
                learning_rate = self.esr_lr_scheduler.get_last_lr()[0]
                self.writer.set_step(log_step)
                self.train_metrics.update('train_bce_loss', reduced_bce_loss.item())
                self.train_metrics.update('train_mse_loss', reduced_mse_loss.item())
                self.train_metrics.update('train_loss', reduced_loss.item())
                self.writer.writer.add_scalar(f'learning rate', learning_rate, global_step=log_step)
                if batch_idx % self.train_log_step == 0:
                    msg = 'Train Epoch: {} {}'.format(epoch,
                                                      self._progress(batch_idx, self.train_dataloader, is_train=True))
                    msg += ' {}: {:.4e}'.format('train_bce_loss', reduced_bce_loss.item())
                    msg += ' {}: {:.4e}'.format('train_mse_loss', reduced_mse_loss.item())
                    msg += ' {}: {:.4e}'.format('train_loss', reduced_loss.item())
                    msg += ' {}: {:.4e}'.format('learning rate', learning_rate)
                    self.logger.debug(msg)

            # Must clear cache at regular interval
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            dist.barrier()

        # only main process has non-zero train_log
        train_log = self.train_metrics.result()

        # do validation
        if self.do_validation:
            if epoch % self.valid_step == 0:
                with torch.no_grad():
                    val_log = self._valid(epoch)
                    train_log.update(val_log)

        # change learning rate
        if self.esr_lr_scheduler is not None:
            self.esr_lr_scheduler.step()

        return train_log

    def _valid(self, stamp):
        """
        Validate after training an epoch or several iterations

        :param stamp: the timestamp for validation,
                      epoch-based training -> epoch; iteration-based training -> valid_stamp
        :return: A log that contains information about validation
        """
        self.logger.debug('validation')

        self.esr_model.eval()

        self.valid_metrics.reset()
        for batch_idx, inputs_seq in enumerate(self.valid_dataloader):

            # forward pass
            loss = 0
            init = True
            for inputs in inputs_seq:
                # forward pass
                input_stack = inputs['inp_cnt'].transpose(1, 2) 

                gt_stack = inputs['gt_cnt'][:, 1].to(self.device)  # Bx2xkHxkW

                if init:
                    init_temp = torch.zeros_like(input_stack[:, 0:1, 0, :, :])
                    init_o = init_temp.repeat(1, self.scale * self.scale * 2, 1, 1).to(self.device)
                    init_h = init_temp.repeat(1, self.n_c, 1, 1).to(self.device)
                    init_h_p = init_temp.repeat(1, self.n_c, 1, 1).to(self.device)
                    init_h_n = init_temp.repeat(1, self.n_c, 1, 1).to(self.device)
                    h, hp, hn, prediction = self.esr_model(input_stack.to(self.device), init_h, init_h_p, init_h_n, init_o, init)
                    init = False
                else:
                    h, hp, hn, prediction = self.esr_model(input_stack.to(self.device), h, hp, hn, prediction, init)

                # loss and backward pass
                if prediction.size()[-2:] != gt_stack.size()[-2:]:
                    scale_prediction = f.interpolate(prediction, size=gt_stack.size()[-2:], mode='bicubic',
                                                     align_corners=False)
                else:
                    scale_prediction = prediction

                mse_loss = self.esr_loss['mse'](scale_prediction, gt_stack)
                loss += mse_loss

            # reduce losses over all GPUs for logging purposes
            reduced_mse_loss = reduce_tensor(mse_loss)
            reduced_loss = reduce_tensor(loss)

            # setup log info
            log_step = (stamp - 1) * len(self.valid_dataloader) + batch_idx
            self.writer.set_step(log_step, 'valid')
            if batch_idx % self.valid_log_step == 0:
                msg = 'Valid timestamp: {} {}'.format(stamp, self._progress(batch_idx, self.valid_dataloader,
                                                                            is_train=False))
                msg += ' {}: {:.4e}'.format('valid_mse_loss', reduced_mse_loss.item())
                msg += ' {}: {:.4e}'.format('valid_loss', reduced_loss.item())
                self.logger.debug(msg)

            self.valid_metrics.update('valid_mse_loss', reduced_mse_loss.item())
            self.valid_metrics.update('valid_loss', reduced_loss.item())

            # Must clear cache at regular interval
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        return self.valid_metrics.result()

    def _save_checkpoint(self, idx, save_best=False):
        """
        Saving checkpoints

        :param idx: epoch-based training -> epoch; iteration-based training -> iteration
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if self.training_mode == 'epoch_based_train':
            state['trainer'] = {
                'training_mode': self.training_mode,
                'epoch': idx,
                'monitor_best': self.mnt_best,
            }
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(idx))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            if save_best:
                best_path = str(self.checkpoint_dir / f'model_best_until_epoch{idx}.pth')
                torch.save(state, best_path)
                self.logger.info(f"Saving current best: model_best_until_epoch{idx}.pth ...")

        elif self.training_mode == 'iteration_based_train':
            filename = str(self.checkpoint_dir / 'checkpoint-iteration{}.pth'.format(idx))
            if save_best:
                best_path = str(self.checkpoint_dir / f'model_best_until_iteration{idx}.pth')
                torch.save(self.esr_model.state_dict(), best_path)
                self.logger.info(f"Saving current best: model_best_until_iteration{idx}.pth ...")
            else:
                torch.save(self.esr_model.state_dict(), filename)
                self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resumer = Resumer(self.config_parser.args.resume, self.logger, self.config_parser.config)
        trainer_states = resumer.resume_trainer('trainer')
        is_same_training_mode = trainer_states['training_mode'] == self.training_mode

        if not self.config_parser.args.reset and is_same_training_mode:
            if self.training_mode == 'epoch_based_train':
                self.start_epoch = trainer_states['epoch'] + 1
                self.mnt_best = trainer_states['monitor_best']
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from epoch {}, \
                                        and use the previous best monitor metric".format(self.start_epoch))

            elif self.training_mode == 'iteration_based_train':
                start_iteration = trainer_states['iteration'] + 1
                self.mnt_best = trainer_states['monitor_best']
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from iteration {}, \
                                        and use the previous best monitor metric".format(start_iteration))

        else:
            if self.training_mode == 'epoch_based_train':
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from epoch 1, \
                                        and reset the previous best monitor metric")

            elif self.training_mode == 'iteration_based_train':
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from iteration 1, \
                                        and reset the previous best monitor metric")

        resumer.resume_model(self.esr_model, 'model')
        resumer.resume_optimizer(self.esr_optimizer, 'optimizer')
        resumer.resume_lr_scheduler(self.esr_lr_scheduler, 'lr_scheduler')

    def _progress(self, idx, data_loader, is_train):
        base = '[{}/{} ({:.0f}%)]'
        current = idx

        if is_train:
            if self.training_mode == 'epoch_based_train':
                total = len(data_loader)
            elif self.training_mode == 'iteration_based_train':
                total = self.iterations
        else:
            total = len(data_loader)

        return base.format(current, total, 100.0 * current / total)


def main(config_parser):
    local_rank = 0

    # fix seed for each process
    seed = config_parser.args.seed
    rank = 0
    init_seeds(seed + rank)

    logger = config_parser.get_logger('train')
    config = config_parser.config
    device = torch.device('cuda:0')
    time_bins = config['TIME_BINS']

    # setup data_loader instances
    train_dataloader = HDF5DataLoaderSequence(config['train_dataloader'])
    valid_dataloader = HDF5DataLoaderSequence(config['valid_dataloader'])


    n_c = 128
    n_b = 5
    model = BMCNet(config['train_dataloader']['dataset']['scale'], n_c, n_b)
    esr_model = model.to(device)

    if rank == 0:
        logger.info(esr_model)

    # loss functions
    esr_loss = {
        'mse': nn.MSELoss(),
    }

    # optimizers
    esr_trainable_params = filter(lambda p: p.requires_grad, esr_model.parameters())
    esr_optimizer = eval(config['optimizer']['name'])(esr_trainable_params, **config['optimizer']['args'])

    # learning rate scheduler
    esr_lr_scheduler = eval(config['lr_scheduler']['name'])(esr_optimizer, **config['lr_scheduler']['args'])

    # training loop
    args = {
        # config parser
        'config_parser': config_parser,
        # dataloader
        'train_dataloader': train_dataloader,
        'valid_dataloader': valid_dataloader,
        # models
        'esr_model': esr_model,
        # loss fts
        'esr_loss': esr_loss,
        # optimizers
        'esr_optimizer': esr_optimizer,
        # lr scheduler
        'esr_lr_scheduler': esr_lr_scheduler,
        # metadata
        'logger': logger,
        'device': device,
        'scale': config['train_dataloader']['dataset']['scale'],
        'n_c': n_c,
    }
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test YAMLParser')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-id', '--runid', default=None, type=str)
    args.add_argument('-seed', '--seed', default=3407, type=int)
    args.add_argument('-r', '--resume', default=None, type=str)
    args.add_argument('--reset', default=False, action='store_true',
                      help='if resume checkpoint, reset trainer states in the checkpoint')
    args.add_argument('--limited_memory', default=False, action='store_true',
                      help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".')

    if args.parse_args().limited_memory:
        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy('file_system')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='test;item1;body1'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config_parser = YAMLParser.from_args(args, options)
    main(config_parser)