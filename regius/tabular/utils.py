# -*- coding: utf-8 -*-
# @Time    : 2020/2/1 下午2:26
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : utils.py
# @Desc    : 表格模型工具类

from ..core import *


def get_embed_col_dims(df: pd.DataFrame, cols: List[str]) -> List[Tuple[str, int]]:
    """获取默认配置下的嵌入维度

    Args:
        df (pd.DataFrame): 原始数据
        cols (List): 需要嵌入的列名

    Returns:
        embed_col_dims (List): 嵌入维度列表
    """
    embed_col_dims = []
    for c in cols:
        n_cat = len(df[c].unique())
        n_dim = min(600, round(1.6 * n_cat ** 0.56))
        embed_col_dims.append((c, n_dim))
    return embed_col_dims


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
