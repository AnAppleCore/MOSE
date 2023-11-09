from .er import *
from .scr import *
from .mose import *


METHODS = {'er': ER, 'scr': SCR, 'mose': MOSE}


def get_agent(method_name, *args, **kwargs):
    if method_name in METHODS.keys():
        return METHODS[method_name](*args, **kwargs)
    else:
        raise Exception('unknown method!')