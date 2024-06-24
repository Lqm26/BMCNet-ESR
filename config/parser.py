import os
import collections
import logging
import argparse
from pathlib import Path
import yaml
from datetime import datetime
from operator import getitem
from functools import reduce
#local modules
from logger import setup_logging


class YAMLParser:
    """ YAML parser for config files """
    def __init__(self, config, args, modification=None):
        # load config file and apply modification
        self._config = self._update_config(config, modification)
        self.args = args
        run_id = args.runid

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['output_path'])

        exper_name = self.config['experiment']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'logs' / exper_name / run_id

        # make directory for saving checkpoints and log.
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        self.log_config(self.config, self.save_dir / 'config.yml')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        assert isinstance(args, argparse.ArgumentParser), 'args is not ArgumentParser class'

        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        cfg_fname = Path(args.config)
        config = cls.parse_config(cfg_fname)

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, cls._get_opt_name(opt.flags)) for opt in options}
        return cls(config, args, modification)
    
    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @staticmethod
    def parse_config(file):
        with open(file) as fid:
            yaml_config = yaml.load(fid, Loader=yaml.FullLoader)
        
        return yaml_config
    
    @staticmethod
    def log_config(config, path_config):
        with open(path_config, "w") as fid:
            yaml.dump(config, fid)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    @staticmethod
    def _get_opt_name(flags):
        for flg in flags:
            if flg.startswith('--'):
                return flg.replace('--', '')
        return flags[0].replace('--', '')

    @classmethod
    def _update_config(cls, config, modification):
        if modification is None:
            return config

        for k, v in modification.items():
            if v is not None:
                cls._set_by_path(config, k, v)
        return config

    @classmethod
    def _set_by_path(cls, tree, keys, value):
        """Set a value in a nested object in tree by sequence of keys."""
        keys = keys.split(';')
        cls._get_by_path(tree, keys[:-1])[keys[-1]] = value

    @classmethod
    def _get_by_path(cls, tree, keys):
        """Access a nested object in tree by sequence of keys."""
        return reduce(getitem, keys, tree)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test YAMLParser')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-id', '--runid', default=None, type=str)
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='test;item1;body1'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config_parser = YAMLParser.from_args(args, options)
    train_logger = config_parser.get_logger('train')
    test_logger = config_parser.get_logger('test')

    pass