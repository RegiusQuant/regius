# -*- coding: utf-8 -*-
# @Time    : 2020/2/1 下午2:26
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : utils.py
# @Desc    : 表格模型工具类

from ..core import *


class MultiColumnLabelEncoder:
    """多列标签编码器,用于神经网络Embedding的输入

    通过使用多个sklearn中的LabelEncoder对每一列进行编码

    Args:
        cols (List): 需要进行编码的列名

    Attributes:
        encoders (Dict): 列名与对应的LabelEncoder字典
    """

    def __init__(self, cols: List[str]):
        self.cols = cols
        self.encoders = {c: LabelEncoder() for c in cols}

    def fit(self, x: pd.DataFrame):
        for c in self.cols:
            self.encoders[c].fit(x[c])
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x_out = x[self.cols].copy()
        for c in self.cols:
            x_out[c] = self.encoders[c].transform(x_out[c])
        return x_out.values

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.fit(x).transform(x)
