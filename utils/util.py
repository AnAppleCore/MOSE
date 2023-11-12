import os
import json
import pickle

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch.distributed as dist
from scipy.stats import sem


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count


class Logger(object):
    def __init__(self, args, base_dir='./outputs'):

        self.args = args
        if args.wandb_log == 'online':
            import wandb
            wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                name    = args.run_name,
                config  = args
            )
            self.wandb = wandb
        else:
            self.wandb = None

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.base_dir = base_dir

        self.folder_path = os.path.join(base_dir, args.run_name+f'_{np.random.randint(1000)}')
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)

        # dump args
        with open(os.path.join(self.folder_path, "params.json"), "wt") as f:
            json.dump(vars(args), f)

        self.to_pickle  = []
        self.picklename = os.path.join(self.folder_path,  "db.pickle")

    def log_scalars(self, values, step, verbose=False):
        for k,v in values.items():
            self.to_pickle += [(k, v, step)]
        if verbose:
            print(values)
        if self.wandb is None:
            return
        self.wandb.log(values, step)

    def log_losses(self, loss_log_holder):
        if self.wandb is None:
            return
        for loss_log in loss_log_holder:
            step = loss_log.pop('step')
            self.log_scalars(loss_log, step, verbose=False)

    def log_accs(self, acc_log):
        step = acc_log.pop('step')
        for k, v in acc_log.items():
            self.to_pickle += [(k, v, step)]
        if self.wandb is None:
            return
        for k, v in acc_log.items():
            for i in range(len(v)):
                self.wandb.log({f"{k}/{i}": v[i]}, step)

    def log_accs_table(self, name, accs_list, step, verbose=False):

        num_tasks, _ = accs_list.shape
        col_name = [f"task{i}" for i in range(num_tasks)]
        accs_table = pd.DataFrame(accs_list, columns=col_name, index=np.array(col_name))
        if verbose:
            print(accs_table)

        accs_table.to_csv(os.path.join(self.folder_path, f"{name}.csv"))

        if self.wandb is None:
            return
        self.wandb.log({name: self.wandb.Table(dataframe=accs_table)}, step)

    def log_img(self, values, step):
        if self.wandb is None:
            return
        for k, fig in values.items():
            figure_path = os.path.join(self.path, f"{k.replace('/', '_')}_step{step}.png")
            fig.savefig(figure_path)
            fig.clear()
            self.wandb.log({k:self.wandb.Image(figure_path)}, step)

    def dump(self):
        f = open(self.picklename, "ab")
        pickle.dump(self.to_pickle, f)
        f.close()
        self.to_pickle = []

    def close(self):
        self.dump()
        if self.wandb is None:
            return
        self.wandb.finish()


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)

    return tensor


def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))


    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt
