import torch
from torch import nn

from src.modeling.utils import set_parameter_requires_grad, RealTimeMetricCalculator


class BaseModel(nn.Module):

    def __init__(
        self, model_fn, weights, n_classes: int, extract_features: bool = False,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.extract_features = extract_features
        self.model = model_fn(weights=weights)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_parameter_requires_grad(model=self.model, extract_features=self.extract_features)

    def predict(self, X):
        return self.model(X)

    def train(self, train_loader, valid_loader, criterion, optimizer, n_epochs: int):
        self.loss_valid_history = []
        self.acc_valid_history = []
        criterion = criterion.to(self.device)
        self.model = self.model.to(self.device)
        for epoch in range(n_epochs):
            train_loss, train_acc = self._train_epoch(
                loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch
            )
            valid_loss, valid_acc = self.evaluate(loader=valid_loader, criterion=criterion)
            self.loss_valid_history.append(valid_loss)
            self.acc_valid_history.append(valid_acc)

    def evaluate(self, loader, criterion):
        self.model.eval()
        eval_acc = RealTimeMetricCalculator()
        eval_loss = RealTimeMetricCalculator()

        with torch.no_grad():
            for (images, labels) in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                examples_in_batch: int = images.size(0)
                outputs = self.predict(images)
                loss = criterion(outputs, labels)
                prediction = outputs.max(1, keepdim=True)[1]
                eval_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / examples_in_batch)
                eval_loss.update(loss.item())
        print(f"MODEL_EVALUATION: acc = {eval_acc.avg} loss = {eval_loss.avg}")
        return eval_loss.avg, eval_acc.avg

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
            outputs = self.predict(images)
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
