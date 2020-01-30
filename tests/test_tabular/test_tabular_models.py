# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())

import torch

from regius.tabular.models import Wide, DeepDense


############################################################
# Wide模块测试
############################################################
def test_wide():
    input_tensor = torch.rand(10, 4)
    model = Wide(in_dim=4, out_dim=1)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([10, 1])


############################################################
# DeepDense模块测试
############################################################
def test_deepdense():
    input_tensor = torch.cat(
        [torch.empty(10, 4).random_(10),
         torch.rand(10, 4)], 1)
    column_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    column_idx = {v: k for k, v in enumerate(column_names)}
    embed_input = [(u, i, j)
                   for u, i, j in zip(column_names[:4], [10] * 4, [5] * 4)]

    model = DeepDense(column_idx=column_idx,
                      hidden_nodes=[16, 8],
                      hidden_drop_ps=[0.2, 0.2],
                      batch_norm=True,
                      embed_input=embed_input,
                      embed_drop_p=0.5,
                      cont_cols=['e', 'f', 'g', 'h'])
    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([10, 8])
