from torch import nn

from .utils import set_parameter_requires_grad


class BaseModel(nn.Module):

    def __init__(self, model_fn, weights, n_classes: int, extract_features: bool = False, use_pretrained: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.extract_features = extract_features
        self.use_pretrained = use_pretrained
        self.model = model_fn(weights=weights)
        set_parameter_requires_grad(model=self.model, extract_features=self.extract_features)
