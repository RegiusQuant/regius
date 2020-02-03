# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import accuracy_score

from regius.tabular.preprocessing import WidePreprocessor, DeepPreprocessor
from regius.tabular.models import Wide, DeepDense, WideDeep
from regius.tabular.learner import WideDeepLearner
from regius.tabular.utils import get_embed_col_dims

if __name__ == '__main__':
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race',
        'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ]
    train_data = pd.read_csv('/media/bnu/data/uci/adult/adult.data', names=column_names)
    print(train_data['education'].unique())

    print('Train Data:')
    print(train_data.head())
    test_data = pd.read_csv('/media/bnu/data/uci/adult/adult.test',
                            names=column_names,
                            skiprows=1)
    print('Test Data:')
    print(test_data.head())

    wide_cols = ['education', 'relationship', 'workclass',
                 'occupation', 'native-country', 'sex']
    wide_prep = WidePreprocessor(wide_cols)
    x_wide = wide_prep.fit_transform(train_data)
    print('-' * 60)
    print('Wide Input Shape:', x_wide.shape)
    print('Wide Feature Names:')
    print(wide_prep.encoder.get_feature_names())

    embed_col_dims = get_embed_col_dims(
        train_data,
        ['education', 'relationship', 'workclass', 'occupation', 'native-country'])
    print('-' * 60)
    print('Embedding Dimension:')
    print(embed_col_dims)

    cont_cols = ['age', 'hours-per-week']
    deep_prep = DeepPreprocessor(embed_col_dims, cont_cols)
    x_deep = deep_prep.fit_transform(train_data)
    print('-' * 60)
    print('Deep Input Shape:', x_deep.shape)
    print('Example Classes (native-country):')
    print(deep_prep.encoder.encoders['native-country'].classes_)

    y = (train_data['income'].apply(lambda x: '>50K' in x)).astype(int).values
    print('-' * 60)
    print('Target Shape:', y.shape)

    wide = Wide(in_dim=x_wide.shape[1], out_dim=1)
    deepdense = DeepDense(
        column_idx=deep_prep.column_idx,
        embed_input=deep_prep.embed_input,
        cont_cols=cont_cols,
        hidden_nodes=[128, 128],
        hidden_drop_ps=[0.2, 0.2],
        batch_norm=True,
        embed_drop_p=0.2
    )
    model = WideDeep(wide=wide, deepdense=deepdense)
    learner = WideDeepLearner(model, objective='binary')
    learner.fit(x_wide, x_deep, y, num_epochs=20, batch_size=128)

    x_wide_test = wide_prep.transform(test_data)
    x_deep_test = deep_prep.transform(test_data)
    y_test = (test_data['income'].apply(lambda x: '>50K' in x)).astype(int).values
    print('-' * 60)
    print('Wide Input Shape (Test):', x_wide_test.shape)
    print('Deep Input Shape (Test):', x_deep_test.shape)
    print('Target Shape (Test):', y_test.shape)

    y_pred = learner.predict(x_wide_test, x_deep_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
