from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

def load_weights(model: nn.Module, path_or_weights, model_prefix: str = None,
                 strict: bool = True) -> nn.Module:
    '''
        This functions loads weights from given path.
        Path variable can be either pathlib.path object or string
        It also handles weights trained in multi-gpu format.
    '''

    if isinstance(path_or_weights, Path):
        path = path_or_weights.resolve().as_posix()
        weights = torch.load(path)
    elif isinstance(path_or_weights, str):
        path = path_or_weights
        weights = torch.load(path)
    elif isinstance(path_or_weights, dict):
        weights = path_or_weights
    else:
        raise NotImplementedError

    new_weights = {}
    for k, v in weights.items():
        if "module." in k:
            new_weights[k.replace("module.", "")] = v
        else:
            new_weights[k] = v

    if(model_prefix is not None):
        weights = new_weights
        new_weights = {}
        for k, v in weights.items():
            orig_prefix = k.split(".")[0]
            new_weights[k.replace(orig_prefix, model_prefix)] = v
        
    model.load_state_dict(new_weights, strict=strict)
    return model