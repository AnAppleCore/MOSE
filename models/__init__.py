from .Resnet18 import resnet18
from .Resnet18_SD import resnet18_sd

METHOD_2_MODEL = {
    'er': resnet18,
    'scr': resnet18, 
    'joint': resnet18,
    'buf': resnet18,
    'mose': resnet18_sd,
    'mose_moe': resnet18_sd
}

def get_model(method_name,  *args, **kwargs):
    if method_name in METHOD_2_MODEL.keys():
        return METHOD_2_MODEL[method_name](*args, **kwargs)
    else:
        raise Exception('unknown method!')