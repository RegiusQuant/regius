# -*- coding: utf-8 -*-
# @Time    : 2020/1/30 上午10:02
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : core.py
# @Desc    : Regius常用类型和函数

import os
from abc import ABCMeta, abstractmethod

# typing相关导入
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

# sklearn相关导入
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

# torch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm, trange

CPU_COUNT = os.cpu_count()
USE_CUDA = torch.cuda.is_available()
