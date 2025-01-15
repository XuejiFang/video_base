import argparse
import importlib
import torch

def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{key: dict_to_namespace(value) for key, value in d.items()})
    else:
        return d

def get_class(config):
    module_name, class_name = config.module, config.class_name
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls

def get_instance(config, **kwargs):
    cls = get_class(config)
    params = getattr(config, "params", {})
    params = vars(params) if isinstance(params, argparse.Namespace) else params
    return cls(**params, **kwargs)

def register_module(config, dtype=torch.float16):
    cls = get_class(config)
    params = getattr(config, "params", {})
    if hasattr(cls, "from_pretrained") and "pretrained" in params:
        return cls.from_pretrained(params.pretrained,  torch_dtype=dtype)
    else:
        raise NotImplementedError("Only support from_pretrained method for now.")

def freeze_module(module: torch.nn.Module, trainable=False) -> torch.nn.Module:
    """Freeze parameters of given module."""
    module.eval() if not trainable else module.train()
    for param in module.parameters():
        param.requires_grad = trainable
    return module
