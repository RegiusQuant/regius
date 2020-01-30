# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())

import pytest
import torch

from regius.tabular.models import Wide, DeepDense

############################################################
# Wide模块测试
############################################################
input_tensor_1 = torch.rand(4, 4)
input_tensor_2 = torch.rand(10, 4)
output_shape_1 = torch.Size([4, 1])
output_shape_2 = torch.Size([10, 1])


@pytest.mark.parametrize('input_tensor, output_shape',
                         [(input_tensor_1, output_shape_1),
                          (input_tensor_2, output_shape_2)])
def test_wide(input_tensor, output_shape):
    model = Wide(in_dim=4, out_dim=1)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == output_shape


############################################################
# DeepDense模块测试
############################################################
x_deepdense = torch.cat([torch.empty(10, 4).random_(10), torch.rand(10, 4)], 1)

column_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
column_idx = {v: k for k, v in enumerate(column_names)}
embed_input = [(u, i, j)
               for u, i, j in zip(column_names[:4], [10] * 4, [5] * 4)]


def test_deepdense():
    model = DeepDense(column_idx=column_idx,
                      hidden_nodes=[16, 8],
                      hidden_drop_ps=[0.2, 0.2],
                      batch_norm=True,
                      embed_input=embed_input,
                      embed_drop_p=0.5,
                      cont_cols=['e', 'f', 'g', 'h'])
    output_tensor = model(x_deepdense)
    assert output_tensor.shape == torch.Size([10, 8])
