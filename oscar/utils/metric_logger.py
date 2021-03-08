# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque
import os

import torch

from .misc import is_main_process


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        # self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        # self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def last_value(self):
        return self.deque[-1]


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.params = {}
        self.delimiter = delimiter

    def update_params(self, update_dict):
        for param_group, group_dict in update_dict.items():
            if param_group not in self.params:
                self.params[param_group] = {}
            for param_name, param_value in group_dict.items():
                # skipping parameters if they start with '_'
                if param_name.startswith('_'):
                    continue
                if isinstance(param_value, torch.Tensor):
                    param_value = param_value.item()
                assert isinstance(param_value, (float, int))
                self.params[param_group][param_name] = param_value

    def update_metrics(self, update_dict):
        for metric_group, group_dict in update_dict.items():
            if metric_group not in self.meters:
                self.meters[metric_group] = defaultdict(SmoothedValue)
            for metric_name, metric_value in group_dict.items():
                # skipping metrics if they start with '_'
                if metric_name.startswith('_'):
                    continue
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                assert isinstance(metric_value, (float, int))
                self.meters[metric_group][metric_name].update(metric_value)

    def get_logs(self, iteration):
        return_str = []
        if len(self.meters) > 0:
            offset_m = max([len(group_name) for group_name in self.meters.keys()])
        else:
            offset_m = 0
        if len(self.params) > 0:
            offset_p = max([len(group_name) for group_name in self.params.keys()])
        else:
            offset_p = 0
        offset = max(offset_m, offset_p)

        for group_name, values in sorted(self.meters.items(),
                                         key=lambda x: x[0]):
            loss_str = []
            for name, meter in values.items():
                loss_str.append("{}: {:.4f} ({:.4f})".format(
                    name, meter.median, meter.global_avg,
                ))
            return_str.append(
                "{:{offset}s} - {}".format(
                    group_name, self.delimiter.join(loss_str), offset=offset,
                ),
            )
        for group_name, values in self.params.items():
            loss_str = []
            for name, param in values.items():
                loss_str.append("{}: {:.6f}".format(name, param))
            return_str.append(
                "{:{offset}s} - {}".format(
                    group_name, self.delimiter.join(loss_str), offset=offset,
                ),
            )
        return "\n    ".join(return_str)


class TensorboardLogger(MetricLogger):
    def __init__(self,
                 log_dir,
                 delimiter='\t'):
        super(TensorboardLogger, self).__init__(delimiter)
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                'To use tensorboard please install tensorboardX '
                '[ pip install tensorboardx ].'
            )
        self.philly_tb_logger = None
        self.philly_tb_logger_avg = None
        self.philly_tb_logger_med = None
        if is_main_process():
            self.tb_logger = SummaryWriter(log_dir)
            self.tb_logger_avg = SummaryWriter(os.path.join(log_dir, 'avg'))
            self.tb_logger_med = SummaryWriter(os.path.join(log_dir, 'med'))
        else:
            self.tb_logger = None
            self.tb_logger_avg = None
            self.tb_logger_med = None

    def get_logs(self, iteration):
        if self.tb_logger:
            for group_name, values in self.meters.items():
                for name, meter in values.items():
                    self.tb_logger.add_scalar(
                        '{}/{}'.format(group_name, name),
                        meter.last_value, iteration,
                    )
                    self.tb_logger_avg.add_scalar(
                        '{}/{}'.format(group_name, name),
                        meter.avg, iteration,
                    )
                    self.tb_logger_med.add_scalar(
                        '{}/{}'.format(group_name, name),
                        meter.median, iteration,
                    )
                    if self.philly_tb_logger:
                        self.philly_tb_logger.add_scalar(
                            '{}/{}'.format(group_name, name),
                            meter.last_value, iteration,
                        )
                        self.philly_tb_logger_avg.add_scalar(
                            '{}/{}'.format(group_name, name),
                            meter.avg, iteration,
                        )
                        self.philly_tb_logger_med.add_scalar(
                            '{}/{}'.format(group_name, name),
                            meter.median, iteration,
                        )
            for group_name, values in self.params.items():
                for name, param in values.items():
                    self.tb_logger.add_scalar(
                        '{}/{}'.format(group_name, name),
                        param, iteration,
                    )
                    if self.philly_tb_logger:
                        self.philly_tb_logger.add_scalar(
                            '{}/{}'.format(group_name, name),
                            param, iteration,
                        )
        return super(TensorboardLogger, self).get_logs(iteration)

    def close(self):
        if is_main_process():
            self.tb_logger.close()
            self.tb_logger_avg.close()
            self.tb_logger_med.close()
