def set_parameter_requires_grad(model, extract_features: bool):
    if extract_features:
        for param in model.parameters():
            param.require_grad = False
