# -*- coding: utf-8 -*-
# @Time    : 2020/2/1 下午12:47
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : preprocessing.py
# @Desc    : Wide&Deep模型预处理

from ..core import *
from .utils import MultiColumnLabelEncoder


class BasePreprocessor:
    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x) -> np.ndarray:
        pass

    def fit_transform(self, x) -> np.ndarray:
        pass


class WidePreprocessor(BasePreprocessor):
    """Wide模块的预处理器

    WidePreprocessor对于类别型变量进行OneHot编码,作为模型Wide部分的输入

    Args:
        wide_cols (List): 包含类别型变量名称的列表

    Attributes:
        encoder (OneHotEncoder): sklearn中的独热编码器
    """

    def __init__(self, wide_cols: List[str]):
        super(WidePreprocessor, self).__init__()
        self.wide_cols = wide_cols
        self.encoder = OneHotEncoder(sparse=False)

    def fit(self, x: pd.DataFrame) -> BasePreprocessor:
        x_wide = x[self.wide_cols].copy()
        self.encoder.fit(x_wide[self.wide_cols])
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x_wide = x[self.wide_cols].copy()
        return self.encoder.transform(x_wide)

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.fit(x).transform(x)


class DeepPreprocessor(BasePreprocessor):
    """DeepDense模块的预处理器

    DeepPreprocessor包含两个部分:
    (1) 对于类别型变量进行标签编码,编码采用了自定义的MultiColumnLabelEncoder
    (2) 对于连续型变量进行标准化

    Args:
        embed_col_dims (List): 包含有(列名,嵌入维度)的列表
        cont_cols (List): 包含连续型变量名称的列表

    Attributes:
        column_idx (Dict): 列名的索引字典
        embed_cols (List): 包含需要嵌入的变量名称的列表
        embed_input (List): 包含有(列名,类型数目,嵌入维度)的元组列表
        encoder (MultiColumnLabelEncoder): 自定义的多列标签编码器
        scaler (StandardScaler): sklearn中的标准化器
    """

    def __init__(self,
                 embed_col_dims: List[Tuple[str, int]] = None,
                 cont_cols: List[str] = None):
        super(DeepPreprocessor, self).__init__()
        self.embed_col_dims = embed_col_dims
        self.cont_cols = cont_cols
        self.column_idx = dict()

        # 处理类别型特征时的属性
        self.embed_dim = None
        self.embed_cols = None
        self.embed_input = None
        self.encoder = None

        # 处理连续型特征时的属性
        self.scaler = None

    def fit(self, x: pd.DataFrame) -> BasePreprocessor:
        # 处理类别型变量
        if self.embed_col_dims is not None:
            self.embed_dim = dict(self.embed_col_dims)
            self.embed_cols = [e[0] for e in self.embed_col_dims]
            self.encoder = MultiColumnLabelEncoder(self.embed_cols)

            self.encoder.fit(x)
            self.embed_input = []
            for k, v in self.encoder.encoders.items():
                self.embed_input.append((k, len(v.classes_), self.embed_dim[k]))

        if self.cont_cols is not None:
            x_cont = x[self.cont_cols].copy()
            self.scaler = StandardScaler().fit(x_cont)

        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x_embed, x_cont = None, None
        if self.embed_col_dims is not None:
            x_embed = self.encoder.transform(x)
        if self.cont_cols is not None:
            x_cont = x[self.cont_cols].copy().values
            x_cont = self.scaler.transform(x_cont)

        if x_embed is not None and x_cont is not None:
            x_out = np.concatenate([x_embed, x_cont], axis=1)
            columns = self.embed_cols + self.cont_cols
        elif x_embed is not None:
            x_out = x_embed
            columns = self.embed_cols
        else:
            x_out = x_cont
            columns = self.cont_cols

        self.column_idx = {v: k for k, v in enumerate(columns)}
        return x_out

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.fit(x).transform(x)
