# -*- coding: utf-8 -*-
# @Time    : 2020/1/29 下午4:42
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : test_tabular_models.py
# @Desc    : 说明

import pytest
import torch

from regius.tabular.models import Wide

input_tensor_1 = torch.rand(4, 4)
input_tensor_2 = torch.rand(10, 4)
output_shape_1 = torch.Size([4, 1])
output_shape_2 = torch.Size([10, 1])


@pytest.mark.parametrize(
    'input_tensor, output_shape',
    [(input_tensor_1, output_shape_1),
     (input_tensor_2, output_shape_2)])
def test_wide(input_tensor, output_shape):
    model = Wide(in_dim=4, out_dim=1)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == output_shape
