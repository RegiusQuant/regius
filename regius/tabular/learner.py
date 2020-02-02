# -*- coding: utf-8 -*-
# @Time    : 2020/2/2 下午1:45
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : learner.py
# @Desc    : Wide&Deep模型训练


from ..core import *
from .data import WideDeepDataset


class WideDeepLearner:
    def __init__(self, model: nn.Module, objective: str):
        self.model = model
        self.objective = objective
        self.optimizer = optim.Adam(self.model.parameters())

        if USE_CUDA:
            self.model.cuda()

        self._temp_loss = 0.

    def _split_dataset(self, x_wide, x_deep, y, test_size=0.2):
        x_train_wide, x_valid_wide, x_train_deep, x_valid_deep, y_train, y_valid = train_test_split(
            x_wide, x_deep, y, test_size=test_size,
            stratify=y if self.objective != 'regression' else None
        )
        x_train = {'x_wide': x_train_wide, 'x_deep': x_train_deep, 'y': y_train}
        x_valid = {'x_wide': x_valid_wide, 'x_deep': x_valid_deep, 'y': y_valid}

        train_dataset = WideDeepDataset(**x_train)
        valid_dataset = WideDeepDataset(**x_valid)
        return train_dataset, valid_dataset

    def _acti_func(self, x: Tensor) -> Tensor:
        if self.objective == 'binary':
            return torch.sigmoid(x)
        else:
            return x

    def _loss_func(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if self.objective == 'binary':
            return F.binary_cross_entropy(y_pred, y_true.view(-1, 1))
        else:
            return F.mse_loss(y_pred, y_true.view(-1, 1))

    def _make_train_step(self, x: Dict[str, Tensor], y: Tensor, batch_idx):
        self.model.train()

        x = {k: v.cuda() for k, v in x.items()} if USE_CUDA else x
        y = y.float()
        y = y.cuda() if USE_CUDA else y

        self.optimizer.zero_grad()
        y_pred = self._acti_func(self.model(x))

        loss = self._loss_func(y_pred, y)
        loss.backward()
        self.optimizer.step()

        self._temp_loss += loss.item()
        train_loss = self._temp_loss / (batch_idx + 1)

        return train_loss

    def fit(self, x_wide, x_deep, y, num_epochs, batch_size):
        train_dataset, valid_dataset = self._split_dataset(x_wide, x_deep, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=CPU_COUNT)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=CPU_COUNT)

        for epoch in range(num_epochs):
            pbar = tqdm(train_loader)
            self._temp_loss = 0.
            for batch_idx, (inputs, targets) in enumerate(pbar):
                pbar.set_description('Epoch %i' % (epoch + 1))
                train_loss = self._make_train_step(inputs, targets, batch_idx)
                pbar.set_postfix(loss=train_loss)
