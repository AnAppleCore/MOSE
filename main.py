import argparse
import random
import numpy as np
import torch
import os
import warnings
from multi_runs import multiple_run

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

def bool2string(s):
    if s in {'True', 'true', 'T', 't'}:
        return True
    elif s in {'False', 'false', 'F', 'f'}:
        return False
    else:
        raise ValueError('Not a valid boolean string')


def get_params():
    parser = argparse.ArgumentParser()
    # experiment related
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, help='(default=%(default)f)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='(default=%(default)s)')
    parser.add_argument('--buffer_size', type=int, default=200, help='(default=%(default)s)')
    parser.add_argument('--buffer_batch_size', type=int, default=64, help='(default=%(default)s)')
    parser.add_argument('--run_nums', type=int, default=10, help='(default=%(default)s)')
    parser.add_argument('--batch_size', type=int, default=10, help='(default=%(default)s)')
    parser.add_argument('--epoch', type=int, default=1, help='(default=%(default)s)')
    parser.add_argument('--ins_t', type=float, default=0.07, help='(default=%(default)s)')
    parser.add_argument('--dist_t', type=float, default=3.0, help='(default=%(default)s)')

    parser.add_argument('--gpu_id', type=int, default=0, help='(default=%(default)s)')
    parser.add_argument('--n_workers', type=int, default=8, help='(default=%(default)s)')

    ## Logging 
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--wandb_project', type=str, default='ocl')
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_log', type=str, default='off', choices=['off', 'online'])
    args = parser.parse_args()
    return args


def main(args):
    torch.cuda.set_device(args.gpu_id)
    args.cuda = torch.cuda.is_available()

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('[CUDA is unavailable]')

    multiple_run(args)


if __name__ == '__main__':
    args = get_params()
    main(args)
