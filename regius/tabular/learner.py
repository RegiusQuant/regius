# -*- coding: utf-8 -*-
# @Time    : 2020/2/2 下午1:45
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : regius
# @File    : learner.py
# @Desc    : Wide&Deep模型训练


from ..core import *
from ..callbacks import CallbackContainer, HistoryCallback
from ..metrics import BinaryAccuracy
from .data import WideDeepDataset
from .utils import MultiColumnLabelEncoder


class WideDeepLearner:
    """Wide&Deep模型训练

    WideDeepLearner类对于Wide&Deep模型进行训练,可以解决回归问题和二分类问题

    Args:
        model (nn.Module): Wide&Deep模型
        objective (str): 训练目标(‘regression’ or 'binary')
        y_range (Tuple): 预测值的范围,当objective=‘regression’时有效,例如：(-10.0, 10.0)
    """

    def __init__(self, model: nn.Module, objective: str, y_range: Optional[Tuple[float, float]] = None):
        if objective not in ['regression', 'binary']:
            raise ValueError('objective must in %s' % (['regression', 'binary']))

        self.model = model
        self.objective = objective
        self.y_range = y_range

        # 设定默认batch_size大小
        self.batch_size = 128
        # 设定默认的优化器
        self.optimizer = optim.AdamW(self.model.parameters())

        # 设定评测指标,二分类问题默认采用准确率
        if self.objective == 'binary':
            self.metric = BinaryAccuracy()
        else:
            self.metric = None

        # 日志记录和回调
        self.history = HistoryCallback()
        self.callback_container = CallbackContainer([self.history])
        self.callback_container.set_model(self.model)

        # 是否使用GPU
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
        """根据预测类型获取输出前的激活函数"""
        if self.objective == 'binary':
            return torch.sigmoid(x)
        else:
            # regression
            if self.y_range is not None:
                # 当设定y_range时,对输出进行sigmoid变换映射到输出区间
                return (self.y_range[1] - self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
            else:
                return x

    def _loss_func(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """根据预测类型获取损失函数"""
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

        if self.metric is not None:
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

        if self.metric is not None:
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

        self.callback_container.on_train_begin(logs={
            'batch_size': self.batch_size,
            'num_epochs': num_epochs,
        })

        for epoch in range(num_epochs):
            train_metric, train_loss, valid_metric, valid_loss = None, None, None, None

            # 模型训练
            if self.metric:
                self.metric.reset()
            self._temp_train_loss = 0.

            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch=epoch, logs=epoch_logs)

            pbar = tqdm(train_loader)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                self.callback_container.on_batch_begin(batch=batch_idx)
                pbar.set_description('Epoch %i' % (epoch + 1))
                train_metric, train_loss = self._make_train_step(inputs, targets, batch_idx)
                if train_metric:
                    pbar.set_postfix(metric=train_metric, loss=train_loss)
                else:
                    pbar.set_postfix(loss=train_loss)
                self.callback_container.on_batch_end(batch=batch_idx)

            if train_metric:
                epoch_logs['train_metric'] = train_metric
            epoch_logs['train_loss'] = train_loss

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

            if valid_metric:
                epoch_logs['valid_metric'] = valid_metric
            epoch_logs['valid_loss'] = valid_loss
            self.callback_container.on_epoch_end(epoch=epoch, logs=epoch_logs)

        self.callback_container.on_train_end()

    def predict(self, x_wide: np.ndarray, x_deep: np.ndarray) -> np.ndarray:
        """进行模型预测

        Args:
            x_wide (np.ndarray): Wide模块输入
            x_deep (np.ndarray): DeepDense模块输入

        Returns:
            y_pred (np.ndarray): 模型预测值
        """
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

    def get_embeddings(self, column_name: str, encoder: MultiColumnLabelEncoder):
        """获取类别型变量的嵌入特征

        Args:
            column_name (str): 类别型变量的列名
            encoder (MultiColumnLabelEncoder): 自定义的多列编码器,DeepPreprocessor中可以获得

        Returns:
            embed_dict (Dict): 对应列的嵌入特征列表
        """
        embed_matrix = None
        for n, p in self.model.named_parameters():
            if 'embed_layers' in n and column_name in n:
                embed_matrix = p.cpu().data.numpy()
                break

        if embed_matrix is None:
            raise ValueError('column_name not in model.')
        embed_dict = {c: embed_matrix[i]
                      for i, c in enumerate(encoder.encoders[column_name].classes_)}
        return embed_dict

    def get_wide_weights(self, wide_features: Optional[List[str]] = None):
        wide_weights = self.model.wide.linear.weight.cpu().data.numpy().squeeze()
        weight_dict = {c: v for c, v in zip(wide_features, wide_weights)}
        return weight_dict
