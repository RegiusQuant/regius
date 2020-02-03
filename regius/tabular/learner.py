# -*- coding: utf-8 -*-
# @Time    : 2020/2/2 下午1:45
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : learner.py
# @Desc    : Wide&Deep模型训练


from ..core import *
from ..metrics import BinaryAccuracy
from .data import WideDeepDataset


class WideDeepLearner:
    """Wide&Deep模型训练

    WideDeepLearner类对于Wide&Deep模型进行训练,可以解决回归问题和二分类问题

    Args:
        model (nn.Module): Wide&Deep模型
        objective (str): 训练目标(‘regression’ or 'binary')
    """

    def __init__(self, model: nn.Module, objective: str):
        self.model = model
        self.objective = objective
        self.optimizer = optim.Adam(self.model.parameters())

        # 设定评测指标,二分类问题默认采用准确率
        if self.objective == 'binary':
            self.metric = BinaryAccuracy()
        else:
            self.metric = None
        # 设定默认batch_size大小
        self.batch_size = 128

        if USE_CUDA:
            self.model.cuda()

        self._temp_train_loss = 0.
        self._temp_valid_loss = 0.

    def _split_dataset(self, x_wide: np.ndarray, x_deep: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """将数据划分为训练集和验证集

        Args:
            x_wide (np.ndarray): Wide模块输入
            x_deep (np.ndarray): DeepDense模块输入
            y (np.ndarray): 预测目标值
            test_size (float): 验证集比例,为0-1之间的浮点数

        Returns:
            train_dataset (WideDeepDataset): 训练集的Dataset
            valid_dataset (WideDeepDataset): 验证集的Dataset
        """
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

    def _make_train_step(self, x: Dict[str, Tensor], y: Tensor, batch_idx: int):
        self.model.train()

        x = {k: v.cuda() for k, v in x.items()} if USE_CUDA else x
        y = y.float()
        y = y.cuda() if USE_CUDA else y

        self.optimizer.zero_grad()
        y_pred = self._acti_func(self.model(x))

        loss = self._loss_func(y_pred, y)
        loss.backward()
        self.optimizer.step()

        self._temp_train_loss += loss.item()
        train_loss = self._temp_train_loss / (batch_idx + 1)

        if self.metric:
            train_metric = self.metric(y_pred, y)
            return train_metric, train_loss
        return None, train_loss

    def _make_valid_step(self, x: Dict[str, Tensor], y: Tensor, batch_idx: int):
        self.model.eval()

        with torch.no_grad():
            x = {k: v.cuda() for k, v in x.items()} if USE_CUDA else x
            y = y.float()
            y = y.cuda() if USE_CUDA else y

            y_pred = self._acti_func(self.model(x))
            loss = self._loss_func(y_pred, y)
            self._temp_valid_loss += loss.item()
            valid_loss = self._temp_valid_loss / (batch_idx + 1)

        if self.metric:
            valid_metric = self.metric(y_pred, y)
            return valid_metric, valid_loss
        return None, valid_loss

    def fit(self,
            x_wide: np.ndarray,
            x_deep: np.ndarray,
            y: np.ndarray,
            num_epochs: int,
            batch_size: int):
        """根据指定的训练轮数和批大小训练模型

        Args:
            x_wide (np.ndarray): Wide模块输入
            x_deep (np.ndarray): DeepDense模块输入
            y (np.ndarray): 预测目标值
            num_epochs (int): 训练轮数
            batch_size (int): 每个Batch的大小
        """
        self.batch_size = batch_size
        train_dataset, valid_dataset = self._split_dataset(x_wide, x_deep, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=CPU_COUNT)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=CPU_COUNT)

        for epoch in range(num_epochs):
            # 模型训练
            if self.metric:
                self.metric.reset()
            self._temp_train_loss = 0.

            pbar = tqdm(train_loader)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                pbar.set_description('Epoch %i' % (epoch + 1))
                train_metric, train_loss = self._make_train_step(inputs, targets, batch_idx)
                if train_metric:
                    pbar.set_postfix(metric=train_metric, loss=train_loss)
                else:
                    pbar.set_postfix(loss=train_loss)

            # 模型验证
            if self.metric:
                self.metric.reset()
            self._temp_valid_loss = 0.

            pbar = tqdm(valid_loader)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                pbar.set_description('Validation')
                valid_metric, valid_loss = self._make_valid_step(inputs, targets, batch_idx)
                if valid_metric:
                    pbar.set_postfix(metric=valid_metric, loss=valid_loss)
                else:
                    pbar.set_postfix(loss=valid_loss)

    def predict(self, x_wide: np.ndarray, x_deep: np.ndarray) -> np.ndarray:
        x_test = {'x_wide': x_wide, 'x_deep': x_deep}
        test_dataset = WideDeepDataset(**x_test)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=CPU_COUNT,
                                 shuffle=False)

        self.model.eval()
        batch_pred_list = []
        with torch.no_grad():
            pbar = tqdm(test_loader)
            for inputs in pbar:
                pbar.set_description('Predict')
                x = {k: v.cuda() for k, v in inputs.items()} if USE_CUDA else inputs
                batch_pred = self._acti_func(self.model(x))
                batch_pred = batch_pred.cpu().data.numpy()
                batch_pred_list.append(batch_pred)
        self.model.train()

        if self.objective == 'regression':
            return np.vstack(batch_pred_list).squeeze(1)
        else:
            y_pred = np.vstack(batch_pred_list).squeeze(1)
            return (y_pred > 0.5).astype('int')
