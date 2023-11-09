import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from losses.loss import sup_con_loss
from utils import get_transform


class SCR(object):
    def __init__(self, model:nn.Module, buffer, optimizer, input_size, args):
        self.model = model
        self.optimizer = optimizer

        self.ins_t = args.ins_t
        self.epoch = args.epoch

        self.buffer = buffer
        self.buffer_per_class = 7
        self.buffer_batch_size = args.buffer_batch_size

        if args.dataset == "cifar10":
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.total_samples = 10000
        self.print_num = self.total_samples // 10

        self.transform = get_transform(args.augmentation, input_size)

        self.total_step = 0
        self.class_holder = []
        self.scaler = GradScaler()

    def train_any_task(self, task_id, train_loader):
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
                'train/ins':      0.,
            }

            if len(self.buffer) > 0:

                with autocast():
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    x = x.requires_grad_()
                    buffer_batch_size = min(
                        self.buffer_batch_size, self.buffer_per_class * len(self.class_holder)
                    )
                    mem_x, mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=None)

                    cat_x = torch.cat((x, mem_x))
                    cat_y = torch.cat((y, mem_y))

                    cat_x_aug = self.transform(cat_x)
                    all_x = torch.cat((cat_x, cat_x_aug))
                    all_y = torch.cat((cat_y, cat_y))

                    feat, proj = self.model(all_x, use_proj=True)
                    ins_loss = sup_con_loss(proj, self.ins_t, all_y)

                    loss += ins_loss
                    loss_log['train/ins'] += ins_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

            loss_log['train/loss'] = loss
            epoch_log_holder.append(loss_log)
            self.total_step += 1

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(f"==>>> it: {batch_idx}, loss: ins {loss_log['train/ins']:.3f} = {loss:.6f}, {100 * (num_d / self.total_samples)}%")

        return epoch_log_holder

    def train(self, task_id, train_loader):
        self.model.train()
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log_holder = self.train_any_task(task_id, train_loader)
            train_log_holder.extend(epoch_log_holder)
            # self.buffer.print_per_task_num()
        return train_log_holder

    def test(self, i, task_loader):
        self.model.eval()
        # calculate the calss means
        print("\nCalculate class means...\n")
        self.class_means = {}
        class_inputs = {cls: [] for cls in self.class_holder}
        for x, y in zip(self.buffer.x, self.buffer.y_int):
            class_inputs[y.item()].append(x)

        for cls, inputs in class_inputs.items():
            features = []
            for ex in inputs:
                feature = self.model.features(ex.unsqueeze(0))
                feature = feature.detach().clone()
                feature = F.normalize(feature, dim=1)
                features.append(feature.squeeze())

            if len(features) == 0:
                mu_y = torch.normal(
                    0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())
                )
                mu_y = mu_y.to(x.device)

            else:
                features = torch.stack(features)
                mu_y = features.mean(0)

            mu_y = F.normalize(mu_y.reshape(1, -1), dim=1)
            self.class_means[cls] = mu_y.squeeze()
        
        all_acc_list = {'step': self.total_step}
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model(task_loader[j]['test'], j)
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

            features = self.model.features(data)
            features = F.normalize(features, dim=1)
            features = features.unsqueeze(2)
            means = torch.stack([self.class_means[cls] for cls in self.class_holder])
            means = torch.stack([means] * data.size(0))
            means = means.transpose(1, 2)
            features = features.expand_as(means)
            dists = (features - means).pow(2).sum(1).squeeze()
            pred = dists.min(1)[1]
            pred = torch.Tensor(self.class_holder)[pred].to(data.device)

            num += data.size()[0]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

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