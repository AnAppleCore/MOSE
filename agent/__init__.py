from .er import *
from .scr import *
from .mose import *
from .joint import *
from .buf import *
from .gsa import *


METHODS = {
    'er': ER,
    'scr': SCR,
    'mose': MOSE,
    'joint': Joint,
    'buf': Buf,
    'gsa': GSA,
}


def get_agent(method_name, *args, **kwargs):
    if method_name in METHODS.keys():
        return METHODS[method_name](*args, **kwargs)
    else:
        raise Exception('unknown method!')