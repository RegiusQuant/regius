# -*- coding: utf-8 -*-
# @Time    : 2020/1/29 下午4:23
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : models.py
# @Desc    : Wide&Deep模型文件

from ..core import *


class Wide(nn.Module):
    """Wide&Deep模型Wide模块

    Wide类将独热编码后的数据通过简单的线性层进行处理

    Args:
        in_dim (int): Wide模块的输入维度
        out_dim (int): Wide模块的输出维度

    Attributes:
        linear (nn.Module): Wide模块的线性层
    """
    def __init__(self, in_dim: int, out_dim: int = 1):
        super(Wide, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.float())


class DeepDense(nn.Module):
    """Wide&Deep模型DeepDense模块

    DeepDense类对类别型变量进行嵌入表示并与连续型变量进行拼接
    拼接后的结果通过一系列的隐藏层输出

    Args:
        column_idx (Dict): 列名索引字典,用于将张量切片
        hidden_nodes (List): 隐藏层节点数量
        hidden_drop_ps (List): 隐藏层Dropout几率
        batch_norm (bool): 隐藏层是否使用Batch Normalization
        embed_input (List): 包含有(列名,类型数目,嵌入维度)的元组列表
        embed_drop_p (float): 嵌入层的Dropout几率
        cont_cols (List): 连续型变量的列名

    Attributes:
        embed_layers (nn.ModuleDict): 嵌入层模块字典
        dense_layers (nn.Sequential): 接受嵌入层输出和连续型变量的隐含层
        out_dim (int): DeepDense模块输出维度
    """
    def __init__(self,
                 column_idx: Dict[str, int],
                 hidden_nodes: List[int],
                 hidden_drop_ps: Optional[List[float]] = None,
                 batch_norm: bool = False,
                 embed_input: Optional[List[Tuple[str, int, int]]] = None,
                 embed_drop_p: float = 0.,
                 cont_cols: Optional[List[str]] = None):
        super(DeepDense, self).__init__()
        self.column_idx = column_idx
        self.batch_norm = batch_norm
        self.embed_input = embed_input
        self.cont_cols = cont_cols

        # 类别型变量
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict({
                'embed_layer_' + col: nn.Embedding(num, dim)
                for col, num, dim in self.embed_input
            })
            self.embed_dropout = nn.Dropout(embed_drop_p)
            embed_in_dim = np.sum([x[2] for x in self.embed_input])
        else:
            embed_in_dim = 0

        # 连续型变量
        cont_in_dim = len(self.cont_cols) if self.cont_cols is not None else 0

        # 构建隐藏层
        in_dim = embed_in_dim + cont_in_dim
        hidden_nodes = [in_dim] + hidden_nodes
        if not hidden_drop_ps:
            hidden_drop_ps = [0.] * (len(hidden_nodes) - 1)
        self.dense_layers = nn.Sequential()
        for i in range(1, len(hidden_nodes)):
            self.dense_layers.add_module(
                'dense_layer_{}'.format(i - 1),
                self._create_dense_layer(
                    hidden_nodes[i - 1],
                    hidden_nodes[i],
                    hidden_drop_ps[i - 1],
                ))
        self.out_dim = hidden_nodes[-1]

    def _create_dense_layer(self,
                            in_dim: int,
                            out_dim: int,
                            drop_p: float = 0.):
        layers = [nn.Linear(in_dim, out_dim), nn.LeakyReLU(inplace=True)]
        if self.batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.Dropout(drop_p))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x_emb, x_cont = None, None

        # 类别型变量
        if self.embed_input is not None:
            x_emb = [
                self.embed_layers['embed_layer_' + col](
                    x[:, self.column_idx[col]].long())
                for col, _, _ in self.embed_input
            ]
            x_emb = torch.cat(x_emb, 1)
            x_emb = self.embed_dropout(x_emb)
        # 连续型变量
        if self.cont_cols is not None:
            cont_idx = [self.column_idx[col] for col in self.cont_cols]
            x_cont = x[:, cont_idx].float()

        if x_emb is None:
            x_out = x_cont
        else:
            x_out = x_emb if x_cont is None else torch.cat([x_emb, x_cont], 1)
        return self.dense_layers(x_out)


class WideDeep(nn.Module):
    """Wide&Deep模型

    Google的Wide&Deep算法实现,用于处理传统表格数据
    论文参考: https://arxiv.org/abs/1606.07792
    实现参考: https://github.com/jrzaurin/pytorch-widedeep

    Args:
        wide (nn.Module): Wide&Deep模型中的Wide模块
        deepdense (nn.Module): Wide&Deep模型中的DeepDense模块
        out_dim (int): 最终输出的维度,'1'对应回归或者二分类问题,'n_class'对应多分类问题
    """
    def __init__(self, wide: nn.Module, deepdense: nn.Module,
                 out_dim: int = 1):
        super(WideDeep, self).__init__()
        self.wide = wide
        self.deepdense = nn.Sequential(deepdense,
                                       nn.Linear(deepdense.out_dim, out_dim))

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            x (Dict): 包含模型名('wide', 'deepdense')与对应张量的字典
        """
        x_out = self.wide(x['wide'])
        x_out.add_(self.deepdense(x['deepdense']))
        return x_out
