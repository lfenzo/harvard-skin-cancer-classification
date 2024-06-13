def set_parameter_requires_grad(model, extract_features: bool):
    if extract_features:
        for param in model.parameters():
            param.require_grad = False


class RealTimeMetricCalculator:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.value = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
