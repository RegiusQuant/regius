# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())

import pandas as pd

from regius.tabular.preprocessing import WidePreprocessor, DeepPreprocessor
from regius.tabular.utils import MultiColumnLabelEncoder


############################################################
# WidePreprocessor模块测试
############################################################

def test_wide_preprocessor():
    x = pd.DataFrame({
        'col1': ['a', 'c', 'c', 'b', 'a'],
        'col2': [1, 4, 7, 7, 2],
        'col3': [1.0, 2.0, 1.1, 1.2, 1.5]
    })

    preprocessor = WidePreprocessor(wide_cols=['col1', 'col2'])
    x_out = preprocessor.fit_transform(x)
    assert x_out.shape == (5, 7)


############################################################
# MultiColumnLabelEncoder模块测试
############################################################
def test_multi_label_encoder():
    x = pd.DataFrame({
        'col1': ['a', 'c', 'c', 'b', 'a'],
        'col2': [1, 4, 7, 7, 2],
        'col3': [1.0, 2.0, 1.1, 1.2, 1.5]
    })

    encoder = MultiColumnLabelEncoder(cols=['col1', 'col2'])
    encoder.fit(x)
    assert len(encoder.encoders['col1'].classes_) == 3
    assert len(encoder.encoders['col2'].classes_) == 4
    x_out = encoder.transform(x)
    assert x_out.shape == (5, 2)


############################################################
# DeepPreprocessor模块测试
############################################################
def test_deep_preprocessor():
    x = pd.DataFrame({
        'col1': ['a', 'c', 'c', 'b', 'a'],
        'col2': [1, 4, 7, 7, 2],
        'col3': [1.0, 2.0, 1.1, 1.2, 1.5],
        'col4': [10, 100, 2, 5, 6],
        'col5': ['abs', 'abs', 'min', 'min', 'abs']
    })

    preprocessor_1 = DeepPreprocessor(
        embed_col_dims=[('col1', 10), ('col2', 10), ('col5', 5)],
        cont_cols=['col3', 'col4']
    )
    x_out = preprocessor_1.fit_transform(x)
    assert x_out.shape == (5, 5)
    assert preprocessor_1.embed_input == [('col1', 3, 10), ('col2', 4, 10), ('col5', 2, 5)]
    assert preprocessor_1.column_idx == {'col1': 0, 'col2': 1, 'col5': 2, 'col3': 3, 'col4': 4}

    preprocessor_2 = DeepPreprocessor(
        embed_col_dims=[('col1', 10), ('col2', 10), ('col5', 5)],
    )
    x_out = preprocessor_2.fit_transform(x)
    assert x_out.shape == (5, 3)

    preprocessor_2 = DeepPreprocessor(
        cont_cols=['col3', 'col4']
    )
    x_out = preprocessor_2.fit_transform(x)
    assert x_out.shape == (5, 2)
