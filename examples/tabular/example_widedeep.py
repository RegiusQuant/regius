# -*- coding: utf-8 -*-

import pandas as pd

from regius.tabular.preprocessing import WidePreprocessor, DeepPreprocessor
from regius.tabular.models import Wide, DeepDense, WideDeep
from regius.tabular.learner import WideDeepLearner

if __name__ == '__main__':
    adult_data = pd.read_csv('/media/bnu/data/uci/adult/adult.data', names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race',
        'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'])
    print(adult_data.head())

    wide_cols = ['education', 'relationship', 'workclass',
                 'occupation', 'native-country', 'sex']
    wide_prep = WidePreprocessor(wide_cols)
    x_wide = wide_prep.fit_transform(adult_data)
    print('-' * 60)
    print('Wide Input Shape:', x_wide.shape)
    print('Wide Feature Names:')
    print(wide_prep.encoder.get_feature_names())

    embed_col_dims = [('education', 10), ('relationship', 10), ('workclass', 10),
                      ('occupation', 10), ('native-country', 10)]
    cont_cols = ['age', 'hours-per-week']
    deep_prep = DeepPreprocessor(embed_col_dims, cont_cols)
    x_deep = deep_prep.fit_transform(adult_data)
    print('-' * 60)
    print('Deep Input Shape:', x_deep.shape)
    print('Example Classes (native-country):')
    print(deep_prep.encoder.encoders['native-country'].classes_)

    y = (adult_data['income'].apply(lambda x: '>50K' in x)).astype(int).values
    print('-' * 60)
    print('Target Shape:', y.shape)

    wide = Wide(in_dim=x_wide.shape[1], out_dim=1)
    deepdense = DeepDense(
        column_idx=deep_prep.column_idx,
        embed_input=deep_prep.embed_input,
        cont_cols=cont_cols,
        hidden_nodes=[64, 64],
    )
    model = WideDeep(wide=wide, deepdense=deepdense)

    learner = WideDeepLearner(model, objective='binary')
    learner.fit(x_wide, x_deep, y, num_epochs=5, batch_size=256)
