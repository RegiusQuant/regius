# -*- coding: utf-8 -*-
# @Time    : 2020/1/30 上午10:02
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : core.py
# @Desc    : Regius常用类型和函数

import os
import time
from copy import deepcopy
from abc import ABCMeta, abstractmethod

# typing相关导入
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

# scikit-learn相关导入
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_COUNT = torch.cuda.device_count()


def get_gpu_memory():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > temp.txt')
    gpu_memory = [int(x.split()[2]) for x in open('temp.txt', 'r').readlines()]
    os.system('rm temp.txt')
    return gpu_memory
