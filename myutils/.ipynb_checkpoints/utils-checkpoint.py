import os
import datetime
from itertools import repeat
import torch
import pandas as pd
import csv
import numpy as np
import torch.distributed as dist
import yaml
from collections import defaultdict


def normalize_tensor(x):
    """
    Compute mean and stddev of the **nonzero** elements of the event tensor
    we do not use PyTorch's default mean() and std() functions since it's faster
    to compute it by hand than applying those funcs to a masked array
    """
    # if (x != 0).sum() != 0:
    #     mean, stddev = x[x != 0].mean(), x[x != 0].std()
    #     x[x != 0] = (x[x != 0] - mean) / stddev
    nonzero = (x != 0)
    num_nonzeros = nonzero.sum()

    if num_nonzeros > 0:
        mean = x.sum() / num_nonzeros
        stddev = torch.sqrt((x ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero.float()
        x = mask * (x - mean) / stddev

    return x


def torch2cv2(image):
    """convert torch tensor to format compatible with cv2.imwrite"""
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy() 
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)

@torch.no_grad()
def reduce_tensor(input_tensor, average=True):
    world_size = 1
    if world_size < 2:
        return input_tensor

    dist.barrier()
    dist.all_reduce(input_tensor)

    if average:
        input_tensor /= world_size 

    return input_tensor


@torch.no_grad()
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    names = []
    values = []
    # sort the keys so that they are consistent across processes
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.barrier()
    dist.all_reduce(values)
    if average:
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricTracker:
    def __init__(self, keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        # self._data = pd.DataFrame(index=keys)
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    i = 0
    for loader in repeat(data_loader):
        loader.sampler.set_epoch(i)
        yield from loader
        i = i + 1


def resume_model(resume_path, model, key, logger):
    resume_path = str(resume_path)
    logger.info("Loading model state: {} from path: {} ... ".format(key, resume_path))
    map_location = torch.device('cpu')
    checkpoint = torch.load(resume_path, map_location=map_location)

    model.load_state_dict(checkpoint[key])

    return model


def resume_optimizer(resume_path, optimizer, key, logger):
    resume_path = str(resume_path)
    logger.info("Loading optimizer state: {} from path: {} ... ".format(key, resume_path))
    map_location = torch.device('cpu')
    checkpoint = torch.load(resume_path, map_location=map_location)

    optimizer.load_state_dict(checkpoint[key])

    return optimizer


class Resumer:
    def __init__(self, resume_path, logger, config):
        self.resume_path = str(resume_path)
        self.logger = logger
        self.config = config
        self.checkpoint = torch.load(self.resume_path, map_location='cpu')

    def resume_model(self, model, key):
        if self.config[key]['name'] != self.checkpoint[key]['name']:
            if dist.get_rank() == 0:
                self.logger.warning('The model arch of checkpoint is different from the training ones. Not resume...')
            return
        if dist.get_rank() == 0:
            self.logger.info("Loading model state: {} from path: {} ... ".format(key, self.resume_path))
        model.module.load_state_dict(self.checkpoint[key]['states'])

    def resume_optimizer(self, optimizer, key):
        if self.config[key]['name'] != self.checkpoint[key]['name']:
            if dist.get_rank() == 0:
                self.logger.warning('The optimizer arch of checkpoint is different from the training ones. Not resume...')
            return
        self.logger.info("Loading optimizer state: {} from path: {} ... ".format(key, self.resume_path))
        optimizer.load_state_dict(self.checkpoint[key]['states'])

    def resume_lr_scheduler(self, lr_scheduler, key):
        if self.config[key]['name'] != self.checkpoint[key]['name']:
            if dist.get_rank() == 0:
                self.logger.warning('The lr_scheduler arch of checkpoint is different from the training ones. Not resume...')
            return
        if dist.get_rank() == 0:
            self.logger.info("Loading lr_scheduler state: {} from path: {} ... ".format(key, self.resume_path))
        lr_scheduler.load_state_dict(self.checkpoint[key]['states'])

    def resume_trainer(self, key):
        if dist.get_rank() == 0:
            self.logger.info("Loading trainer state: {} from path: {} ... ".format(key, self.resume_path))

        return self.checkpoint[key]


class Logger_yaml():
    def __init__(self, path):
        self.log_file = open(path, 'w')
        self.info_dict = defaultdict(list)

    def log_info(self, info: str):
        self.info_dict['info'].append(info)

    def log_dict(self, dict: dict, name: str):
        self.info_dict[name] = dict

    def __del__(self):
        yaml.dump(dict(self.info_dict), self.log_file)


# class Logger():
#     def __init__(self, path):
#         self.log_file = open(path, 'w', newline='')
#         self.logger = csv.writer(self.log_file, delimiter='\t')

#     def __del__(self):
#         self.log_file.close()

#     def log_info(self, info: list):
#         write_values = []
#         write_values += info

#         self.logger.writerow(write_values)
#         self.log_file.flush()

#     def log_dict(self, dict: dict):
#         for k, v in dict.items():
#             write_values = []
#             if isinstance(v, str):
#                 write_values.append('{:s}: {}'.format(k, v))
#             else:
#                 write_values.append('{:s}: {:.4f}'.format(k, v))

#             self.logger.writerow(write_values)
#             self.log_file.flush()

#     def log_config(self, dict: dict):
#         for k, v in dict.items():
#             write_values = []
#             write_values.append('{:s}: {}'.format(k, v))

#             self.logger.writerow(write_values)
#             self.log_file.flush()


#********************************************************
def load_model(model_dir, model, device):
    """
    Load model from file.
    :param model_dir: model directory
    :param model: instance of the model class to be loaded
    :param device: model device
    :return loaded model
    """

    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device)
        if "state_dict" in model_loaded.keys():
            model_loaded = model_loaded["state_dict"]
        model.load_state_dict(model_loaded)
        print("Model restored from " + model_dir + "\n")

    elif os.path.isdir(model_dir):
        model_name = model_dir + model.__class__.__name__

        extensions = [".pt", ".pth.tar", ".pwf", "_weights_min.pwf"]  # backwards compatibility
        for ext in extensions:
            if os.path.isfile(model_name + ext):
                model_name += ext
                break

        if os.path.isfile(model_name):
            model_loaded = torch.load(model_name, map_location=device)
            if "state_dict" in model_loaded.keys():
                model_loaded = model_loaded["state_dict"]
            model.load_state_dict(model_loaded)
            print("Model restored from " + model_name + "\n")
        else:
            print("No model found at" + model_name + "\n")

    return model


def create_model_dir(path_models, runid):
    """
    Create directory for storing model parameters.
    :param path_models: path in which the model should be stored
    :param runid: MLFlow's unique ID of the model
    :return path to generated model directory
    """

    now = datetime.datetime.now()

    path_models += "model_"
    path_models += "%02d%02d%04d" % (now.day, now.month, now.year)
    path_models += "_%02d%02d%02d_" % (now.hour, now.minute, now.second)
    path_models += runid  # mlflow run ID
    path_models += "/"
    if not os.path.exists(path_models):
        os.makedirs(path_models)
    print("Weights stored at " + path_models + "\n")
    return path_models


def save_model(path_models, model):
    """
    Overwrite previously saved model with new parameters.
    :param path_models: model directory
    :param model: instance of the model class to be saved
    """

    os.system("rm -rf " + path_models + model.__class__.__name__ + ".pt")
    model_name = path_models + model.__class__.__name__ + ".pt"
    torch.save(model.state_dict(), model_name)
