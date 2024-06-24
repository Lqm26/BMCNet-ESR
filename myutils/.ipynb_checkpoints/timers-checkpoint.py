import torch
import torch.distributed as dist
import time
import numpy as np
import atexit
from collections import defaultdict


cuda_timers = defaultdict(list)
timers = defaultdict(list)


# class CudaTimer:
#     def __init__(self, timer_name=''):
#         self.timer_name = timer_name

#         self.start = torch.cuda.Event(enable_timing=True)
#         self.end = torch.cuda.Event(enable_timing=True)

#     def __enter__(self):
#         self.start.record()
#         return self

#     def __exit__(self, *args):
#         self.end.record()
#         torch.cuda.synchronize()
#         cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))

class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        self.interval *= 1000.0  # convert to milliseconds
        timers[self.timer_name].append(self.interval)

class Timer:
    def __init__(self, timer_name='', logger=None):
        self.timer_name = timer_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        self.interval *= 1000.0  # convert to milliseconds
        timers[self.timer_name].append(self.interval)

        if self.timer_name == 'Time of training one epoch' and self.logger != None:
            if dist.get_rank() == 0:
                if self.interval < 1000.0:
                    self.logger.info('{}: {:.2f} ms'.format(self.timer_name, self.interval))
                else:
                    self.logger.info('{}: {:.2f} s'.format(self.timer_name, self.interval / 1000.0))


def print_timing_info():
    print('== Timing statistics ==')
    for timer_name, timing_values in [*cuda_timers.items(), *timers.items()]:
        timing_value = np.mean(np.array(timing_values))
        if timing_value < 1000.0:
            print('{}: {:.2f} ms ({} samples)'.format(timer_name, timing_value, len(timing_values)))
        else:
            print('{}: {:.2f} s ({} samples)'.format(timer_name, timing_value / 1000.0, len(timing_values)))


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
