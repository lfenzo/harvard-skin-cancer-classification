import torch
from torch import nn

from src.modeling.utils import set_parameter_requires_grad, RealTimeMetricCalculator


class BaseModel(nn.Module):

    def __init__(
        self, model_fn, weights, n_classes: int, extract_features: bool = False,
        use_pretrained: bool = True
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.extract_features = extract_features
        self.use_pretrained = use_pretrained
        self.model = model_fn(weights=weights)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_parameter_requires_grad(model=self.model, extract_features=self.extract_features)

    def train(self, train_loader, valid_loader, criterion, optimizer, n_epochs: int):
        self._set_validation_history()
        self._set_training_history()
        criterion = criterion.to(self.device)
        self.model = self.model.to(self.device)
        for epoch in range(n_epochs):
            train_loss, train_acc = self._train_epoch(
                loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch
            )
            valid_loss, valid_acc = self._evaluate_epoch(
                loader=valid_loader, criterion=criterion, optimizer=optimizer, epoch=epoch
            )
            self.loss_valid_history.append(valid_loss)
            self.acc_valid_history.append(valid_acc)

    def evaluate(self, test_loader, criterion):
        pass

    def _set_validation_history(self):
        self.loss_valid_history = []
        self.acc_valid_history = []

    def _set_training_history(self):
        self.loss_train_history = []
        self.acc_train_history = []

    def _evaluate_epoch(self, loader, criterion, optimizer, epoch):
        self.model.eval()
        valid_acc = RealTimeMetricCalculator()
        valid_loss = RealTimeMetricCalculator()

        with torch.no_grad():
            for (images, labels) in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                examples_in_batch: int = images.size(0)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                prediction = outputs.max(1, keepdim=True)[1]
                valid_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / examples_in_batch)
                valid_loss.update(loss.item())
        print(f"EPOCH VALIDATION: [epoch = {epoch}] acc = {valid_acc.avg} loss = {valid_loss.avg}")
        return valid_loss.avg, valid_acc.avg

    def _train_epoch(self, loader, criterion, optimizer, epoch: int):
        self.model.train()
        train_acc = RealTimeMetricCalculator()
        train_loss = RealTimeMetricCalculator()

        global_iteration: int = (epoch - 1) * len(loader)

        for i, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            examples_in_batch: int = images.size(0)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            prediction = outputs.max(1, keepdim=True)[1]

            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / examples_in_batch)
            train_loss.update(loss.item())

            global_iteration += 1

            if (i + 1) % 100 == 0:
                print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                    epoch,
                    i + 1,
                    len(loader),
                    train_loss.avg,
                    train_acc.avg)
                )
        return train_loss.avg, train_acc.avg
