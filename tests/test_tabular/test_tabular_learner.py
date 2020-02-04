# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from regius.tabular.preprocessing import WidePreprocessor, DeepPreprocessor
from regius.tabular.models import Wide, DeepDense, WideDeep
from regius.tabular.learner import WideDeepLearner
from regius.tabular.utils import get_embed_col_dims


def create_test_dataframe(num_samples, objective='regression'):
    if objective == 'regression':
        y = np.random.uniform(-1, 1, (num_samples,))
    else:
        y = np.random.randint(0, 2, (num_samples,))

    train_data = pd.DataFrame({
        'a': np.random.randint(10, 20, (num_samples,)),
        'b': np.random.randint(5, 10, (num_samples,)),
        'c': np.random.normal(0, 1, (num_samples,)),
        'd': np.random.uniform(1, 10, (num_samples,)),
        'y': y
    })
    return train_data


############################################################
# Wide&Deep模型训练测试(Regression)
############################################################
def test_widedeep_regression():
    train_data = create_test_dataframe(num_samples=64, objective='regression')

    wide_cols = ['a', 'b']
    wide_prep = WidePreprocessor(wide_cols)
    x_wide = wide_prep.fit_transform(train_data)

    embed_col_dims = get_embed_col_dims(train_data, ['a', 'b'])
    cont_cols = ['c', 'd']
    deep_prep = DeepPreprocessor(embed_col_dims, cont_cols)
    x_deep = deep_prep.fit_transform(train_data)

    y = train_data['y'].values

    wide = Wide(in_dim=x_wide.shape[1], out_dim=1)
    deepdense = DeepDense(
        column_idx=deep_prep.column_idx,
        embed_input=deep_prep.embed_input,
        cont_cols=cont_cols,
        hidden_nodes=[16, 16],
        hidden_drop_ps=[0.2, 0.2],
        batch_norm=True,
        embed_drop_p=0.2
    )
    model = WideDeep(wide=wide, deepdense=deepdense)
    learner = WideDeepLearner(model, objective='regression', y_range=(-1, 1))
    learner.fit(x_wide, x_deep, y, num_epochs=20, batch_size=16)

    y_pred = learner.predict(x_wide, x_deep)
    print('MSE:', mean_squared_error(y, y_pred))
    print('R2:', r2_score(y, y_pred))


############################################################
# # Wide&Deep模型训练测试(Binary)
############################################################
def test_widedeep_binary():
    train_data = create_test_dataframe(num_samples=64, objective='binary')

    wide_cols = ['a', 'b']
    wide_prep = WidePreprocessor(wide_cols)
    x_wide = wide_prep.fit_transform(train_data)

    embed_col_dims = get_embed_col_dims(train_data, ['a', 'b'])
    cont_cols = ['c', 'd']
    deep_prep = DeepPreprocessor(embed_col_dims, cont_cols)
    x_deep = deep_prep.fit_transform(train_data)

    y = train_data['y'].values

    wide = Wide(in_dim=x_wide.shape[1], out_dim=1)
    deepdense = DeepDense(
        column_idx=deep_prep.column_idx,
        embed_input=deep_prep.embed_input,
        cont_cols=cont_cols,
        hidden_nodes=[16, 16],
        hidden_drop_ps=[0.2, 0.2],
        batch_norm=True,
        embed_drop_p=0.2
    )
    model = WideDeep(wide=wide, deepdense=deepdense)
    learner = WideDeepLearner(model, objective='binary')
    learner.fit(x_wide, x_deep, y, num_epochs=20, batch_size=16)

    y_pred = learner.predict(x_wide, x_deep)
    print('Accuracy:', accuracy_score(y, y_pred))
