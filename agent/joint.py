import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from utils import get_transform


class Joint(object):
    def __init__(self, model:nn.Module, buffer, optimizer, input_size, args):
        self.model = model
        self.optimizer = optimizer

        self.epoch = args.epoch

        self.buffer = buffer
        self.buffer_per_class = 7
        self.buffer_batch_size = args.buffer_batch_size

        if args.dataset == "cifar10":
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.total_samples = 1000
        self.print_num = self.total_samples // 10
        
        self.transform = get_transform(args.augmentation, input_size)

        self.total_step = 0
        self.class_holder = []
        self.scaler = GradScaler()

    def train_any_task(self, task_id, train_loader, epoch):
        num_d = 0
        epoch_log_holder = []
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach().item())

            loss = 0.
            loss_log = {
                'step':     self.total_step,
                'train/loss':     0.,
                'train/ce':       0.,
            }

            with autocast():
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                x = x.detach()
                y = y.detach()

                if self.transform is not None:
                    x_aug = self.transform(x)
                    x_aug = x_aug.detach()
                    pred = self.model(x_aug)
                else:
                    pred = self.model(x)

                ce_loss = F.cross_entropy(pred, y)
                loss = ce_loss
                loss_log['train/ce'] += ce_loss.item()
            
            self.scaler.scale(ce_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if epoch == 0:
                self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

            loss_log['train/loss'] = loss.item()
            epoch_log_holder.append(loss_log)
            self.total_step += 1

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(f"==>>> it: {batch_idx}, loss: ce {loss_log['train/ce']:.3f} = {loss:.6f}, {100 * (num_d / self.total_samples)}%")

        return epoch_log_holder

    def train(self, task_id, train_loader):
        self.model.train()
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log_holder = self.train_any_task(task_id, train_loader, epoch)
            train_log_holder.extend(epoch_log_holder)
            # self.buffer.print_per_task_num()
        return train_log_holder

    def test(self, i, task_loader):
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        with torch.no_grad():
            acc_list = np.zeros(1)
            for j in range(i + 1):
                acc = self.test_model(task_loader['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['3'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def test_model(self, loader, i):
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()

            pred = self.model(data)
            pred = pred.data.max(1, keepdim=True)[1]

            num += data.size()[0]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_buffer(self, i, task_loader):
        # buffer test accuracy of tasks j <= i @ task i
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        with torch.no_grad():
            acc_list = np.zeros(1)
            for j in range(i + 1):
                acc = self.test_buffer_task(j)
                acc_list[j] = acc.item()

            all_acc_list['3'] = acc_list
            print(f"buffer tasks acc:{acc_list}")
            print(f"bufeer tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def test_buffer_task(self, i):
        # buffer test accuracy of tasks i
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        x_i, y_i, _ = self.buffer.onlysample(self.buffer.current_index, task=i)

        for x, y in zip(x_i, y_i):
            x = x.unsqueeze(0).detach()
            y = y.unsqueeze(0).detach()

            pred = self.model(x)
            pred = pred.data.max(1, keepdim=True)[1]

            num += x.size()[0]
            correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_train(self, i, task_loader):
        # train accuracy of current task i
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        with torch.no_grad():
            acc_list = np.zeros(1)
            acc = self.test_model(task_loader['train'], i)
            acc_list[i] = acc.item()

            all_acc_list['3'] = acc_list
            print(f"train tasks acc:{acc_list}")
            print(f"train tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def save_checkpoint(self, save_path = './outputs/final.pt'):
        print(f"Save checkpoint to: {save_path}")
        ckpt_dict = {
            'model': self.model.state_dict(),
            'buffer': self.buffer.state_dict(),
        }
        folder, file_name = os.path.split(save_path)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(ckpt_dict, save_path)

    def load_checkpoint(self, load_path = './outputs/final.pt'):
        print(f"Load checkpoint from: {load_path}")
        ckpt_dict = torch.load(load_path)
        self.model.load_state_dict(ckpt_dict['model'])
        self.buffer.load_state_dict(ckpt_dict['buffer'])
