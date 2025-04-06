import argparse
import os
import random
import warnings

import numpy as np
import torch

from agent import METHODS
from experiment.dataset import DATASETS
from multi_runs import multiple_run
from multi_runs_joint import multiple_run_joint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')


def get_params():
    parser = argparse.ArgumentParser()
    # experiment related
    parser.add_argument('--dataset',            default='cifar10',  type=str, choices=DATASETS.keys())
    parser.add_argument('--n_tasks',            default='10',       type=int)
    parser.add_argument('--buffer_size',        default=200,        type=int)
    parser.add_argument('--method',             default='mose',     type=str, choices=METHODS.keys())

    parser.add_argument('--seed',               default=0,          type=int)
    parser.add_argument('--run_nums',           default=10,         type=int)
    parser.add_argument('--epoch',              default=1,          type=int)
    parser.add_argument('--lr',                 default=1e-3,       type=float)
    parser.add_argument('--wd',                 default=1e-4,       type=float)
    parser.add_argument('--batch_size',         default=10,         type=int)
    parser.add_argument('--buffer_batch_size',  default=64,         type=int)

    parser.add_argument('--continual',          default='on',       type=str, choices=['off', 'on'])

    # mose control
    parser.add_argument('--ins_t',              default=0.07,       type=float)
    parser.add_argument('--expert',             default='3',        type=str, choices=['0','1','2','3'])
    parser.add_argument('--n_experts',          default=4,          type=int)
    parser.add_argument('--classifier',         default='ncm',      type=str, choices=['linear', 'ncm'])
    parser.add_argument('--augmentation',       default='ocm',      type=str, choices=['ocm', 'scr', 'none', 'simclr', 'randaug', 'trivial'])

    parser.add_argument('--gpu_id',             default=0,          type=int)
    parser.add_argument('--n_workers',          default=8,          type=int)

    # logging 
    parser.add_argument('--exp_name',           default='tmp',      type=str)
    parser.add_argument('--wandb_project',      default='ocl',      type=str)
    parser.add_argument('--wandb_entity',                           type=str)
    parser.add_argument('--wandb_log',          default='off',      type=str, choices=['off', 'online'])
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

    if args.continual == 'on':
        multiple_run(args)
    else:
        multiple_run_joint(args)


if __name__ == '__main__':
    args = get_params()
    main(args)
