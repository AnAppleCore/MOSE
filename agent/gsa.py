import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from utils import get_transform


class GSA(object):
    """GSA-lite agent integrated into MOSE.

    - Backbone: any classifier with standard forward(x)->logits over args.n_classes
    - Buffer: models.buffer.Buffer (stores x, y_int, t)
    - Idea: use per-class statistics on replay samples' predicted probabilities
      to reweight the replay cross-entropy, similar in spirit to original GSA.
    """

    def __init__(self, model: nn.Module, buffer, optimizer, input_size, args):
        self.model = model
        self.optimizer = optimizer

        self.epoch = args.epoch
        self.n_classes = args.n_classes
        self.buffer = buffer
        self.buffer_per_class = 7
        self.buffer_batch_size = args.buffer_batch_size

        # dataset sizes for printing progress; follow ER/SCR conventions
        if args.dataset == "cifar10":
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.total_samples = 1000
        elif args.dataset == "mnist":
            self.total_samples = 6000
        else:
            # best-effort default
            self.total_samples = 5000
        self.print_num = max(1, self.total_samples // 10)

        self.transform = get_transform(args.augmentation, input_size)

        self.total_step = 0
        self.class_holder = []  # seen class ids
        self.scaler = GradScaler()

        # for GSA-style statistics over replay predictions
        self.class_per_task = None  # filled lazily from args in first train call
        self.pos_prob_sum = None
        self.neg_prob_sum = None
        self.class_counts = None

    def _ensure_stats(self):
        if self.pos_prob_sum is None:
            self.pos_prob_sum = torch.zeros(self.n_classes).cuda()
            self.neg_prob_sum = torch.zeros(self.n_classes).cuda()
            self.class_counts = torch.zeros(self.n_classes).cuda()

    def _update_class_stats(self, probs, labels):
        """Update per-class positive/negative prob sums and counts.

        probs: [B, C] after softmax
        labels: [B] long, in [0, C-1]
        """
        self._ensure_stats()
        with torch.no_grad():
            B, C = probs.shape
            labels = labels.view(-1)
            # one-hot for labels
            one_hot = torch.zeros_like(probs)
            one_hot[torch.arange(B, device=probs.device), labels] = 1.0

            pos = probs * one_hot
            neg = probs * (1.0 - one_hot)

            self.pos_prob_sum += pos.sum(dim=0)
            self.neg_prob_sum += neg.sum(dim=0)
            self.class_counts += one_hot.sum(dim=0)

    def _compute_replay_loss(self, logits, labels):
        """GSA-inspired weighted cross-entropy on replay samples.

        logits: [B, C], labels: [B]
        """
        self._ensure_stats()
        probs = torch.softmax(logits, dim=1)
        self._update_class_stats(probs.detach(), labels)

        eps = 1e-6
        # ANT like in original code: larger when negative mass is small
        ANT = (self.class_counts - self.pos_prob_sum) / (self.neg_prob_sum + eps)
        # map to (0, 2] as in 2/(1+exp(1-ANT))
        weights_c = 2.0 / (1.0 + torch.exp(1.0 - ANT))  # [C]

        # sample-wise weights
        sample_weights = weights_c[labels]
        log_probs = torch.log_softmax(logits, dim=1)
        nll = -log_probs[torch.arange(logits.size(0), device=logits.device), labels]
        loss = (sample_weights * nll).mean()
        return loss

    def train_any_task(self, task_id, train_loader, epoch):
        num_d = 0
        epoch_log_holder = []

        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach().item())

            loss = 0.0
            loss_log = {
                'step': self.total_step,
                'train/loss': 0.0,
                'train/ce': 0.0,
                'train/replay': 0.0,
            }

            with autocast():
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                cur_x = x.detach()
                cur_y = y.detach()

                if len(self.buffer) > 0:
                    # how many replay samples
                    buffer_batch_size = min(
                        self.buffer_batch_size,
                        self.buffer_per_class * max(1, len(self.class_holder)),
                    )
                    mem_x, mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=None)
                    mem_x = mem_x.detach()
                    mem_y = mem_y.detach()

                    # concat current and replay for CE
                    all_x = torch.cat([cur_x, mem_x], dim=0)
                    all_y = torch.cat([cur_y, mem_y], dim=0)
                else:
                    all_x = cur_x
                    all_y = cur_y

                if self.transform is not None:
                    all_x_aug = self.transform(all_x)
                    all_x_aug = all_x_aug.detach()
                    logits_all = self.model(all_x_aug)
                else:
                    logits_all = self.model(all_x)

                ce_loss = F.cross_entropy(logits_all, all_y)
                loss_log['train/ce'] += ce_loss.item()
                loss = ce_loss

                if len(self.buffer) > 0:
                    # use only replay part for GSA-style weighted loss
                    replay_logits = logits_all[cur_x.size(0):]
                    replay_labels = all_y[cur_x.size(0):]
                    if replay_logits.size(0) > 0:
                        replay_loss = self._compute_replay_loss(replay_logits, replay_labels)
                        loss = loss + 2.0 * replay_loss
                        loss_log['train/replay'] += replay_loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if epoch == 0:
                self.buffer.add_reservoir(x=cur_x.detach(), y=cur_y.detach(), logits=None, t=task_id)

            loss_log['train/loss'] = float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss)
            epoch_log_holder.append(loss_log)
            self.total_step += 1

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(f"==>>> it: {batch_idx}, loss: ce {loss_log['train/ce']:.3f} + replay {loss_log['train/replay']:.3f} = {loss:.6f}, {100 * (num_d / self.total_samples)}%")

        return epoch_log_holder

    def train(self, task_id, train_loader):
        self.model.train()
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log_holder = self.train_any_task(task_id, train_loader, epoch)
            train_log_holder.extend(epoch_log_holder)
        return train_log_holder

    def test(self, i, task_loader):
        """Standard classifier-head evaluation over all seen tasks j<=i.
        This matches ER's behaviour, for fair comparison.
        """
        self.model.eval()
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

    def test_model(self, loader, task_id):
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()

            logits = self.model(data)
            pred = logits.data.max(1, keepdim=True)[1]

            num += data.size()[0]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print(f'Test task {task_id}: Accuracy: {correct}/{num} ({test_accuracy:.2f}%)')
        return test_accuracy

    def save_checkpoint(self, save_path='./outputs/final_gsa.pt'):
        print(f"Save checkpoint to: {save_path}")
        ckpt_dict = {
            'model': self.model.state_dict(),
            'buffer': self.buffer.state_dict(),
        }
        folder, _ = os.path.split(save_path)
        if folder and not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(ckpt_dict, save_path)

    def load_checkpoint(self, load_path='./outputs/final_gsa.pt'):
        print(f"Load checkpoint from: {load_path}")
        ckpt_dict = torch.load(load_path)
        self.model.load_state_dict(ckpt_dict['model'])
        self.buffer.load_state_dict(ckpt_dict['buffer'])

