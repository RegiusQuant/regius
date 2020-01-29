# -*- coding: utf-8 -*-
# @Time    : 2020/1/29 下午4:23
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : models.py
# @Desc    : Wide&Deep模型文件

import torch.nn as nn
from torch import Tensor


class Wide(nn.Module):
    r"""Wide&Deep模型Wide模块

    Wide模块将One-Hot编码后的数据通过简单的线性层进行处理

    Args:
        in_dim: Wide部分的输入维度
        out_dim: Wide部分的输出维度

    Attributes:
        linear: Wide部分的线性层
    """
    def __init__(self, in_dim: int, out_dim: int = 1):
        super(Wide, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.float())
