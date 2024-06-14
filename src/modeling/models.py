from torch import nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    inception_v3, Inception_V3_Weights,
)

from .base_model import BaseModel


class ResNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model_fn=resnet50, weights=ResNet50_Weights.DEFAULT, **kwargs)
        self.input_size = 224
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.n_classes)


class DenseNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model_fn=densenet121, weights=DenseNet121_Weights.DEFAULT, **kwargs)
        self.input_size = 224
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.n_classes)


class Inception(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model_fn=inception_v3, weights=Inception_V3_Weights.DEFAULT, **kwargs)
        self.input_size = 299
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.n_classes)

    def predict(self, X):
        predictions = self.model(X)[0] if self.model.training else self.model(X)
        return predictions
