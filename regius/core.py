# -*- coding: utf-8 -*-
# @Time    : 2020/1/30 上午10:02
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : core.py
# @Desc    : Regius常用类型和函数

# typing相关引用
from typing import Tuple, List, Dict, Optional

import numpy as np
from sklearn.utils import Bunch

# torch相关引用
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader