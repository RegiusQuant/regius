# -*- coding: utf-8 -*-
# @Time    : 2020/1/30 下午2:33
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : data.py
# @Desc    : 表格数据所需的Dataset


from ..core import *


class WideDeepDataset(Dataset):
    """Wide&Deep模型所需的数据集结构

    Args:
        x_wide (np.ndarray): Wide模块的输入特征
        x_deep (np.ndarray): DeepDense模块的输入特征
        y (np.ndarray): 预测目标值
    """
    def __init__(self,
                 x_wide: np.ndarray,
                 x_deep: np.ndarray,
                 y: Optional[np.ndarray] = None):
        self.x_wide = x_wide
        self.x_deep = x_deep
        self.y = y

    def __getitem__(self, idx: int):
        x = Bunch(wide=self.x_wide[idx], deep=self.x_deep[idx])
        if self.y is not None:
            return x, self.y[idx]
        return x

    def __len__(self):
        return len(self.x_deep)
