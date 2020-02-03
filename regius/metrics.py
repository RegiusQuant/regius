# -*- coding: utf-8 -*-
# @Time    : 2020/2/3 下午1:40
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : metrics.py
# @Desc    : 自定义评估指标

from .core import *


class BaseMetric(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        raise NotImplementedError('自定义指标必须实现该方法')

    @abstractmethod
    def __call__(self, y_pred: Tensor, y_true: Tensor):
        raise NotImplementedError('自定义指标必须实现该方法')


class BinaryAccuracy(BaseMetric):

    def __init__(self):
        super(BinaryAccuracy, self).__init__()
        self.correct_count = 0
        self.total_count = 0

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        self.correct_count += y_pred.round().eq(y_true.view(-1, 1)).float().sum().item()
        self.total_count += len(y_pred)
        acc = float(self.correct_count) / float(self.total_count)
        return np.round(acc, 4)
