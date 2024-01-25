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
from utils.rotation_transform import RandomFlip


class MOSE_MOE(object):
    def __init__(self, model:nn.Module, buffer, optimizer, input_size, args):
        self.model = model
        self.optimizer = optimizer

        self.ins_t = args.ins_t
        self.epoch = args.epoch
        self.expert = int(args.expert)
        self.n_classes_num = args.n_classes
        self.use_ncm = (args.classifier == 'ncm')

        self.buffer = buffer
        self.buffer_per_class = 7
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_cur_task = (self.buffer_batch_size // 2) - args.batch_size

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
        if epoch == 0:
            self.new_class_holder = []
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach().item())
                    self.new_class_holder.append(Y[j].detach().item())

            loss = 0.
            loss_log = {
                'step':     self.total_step,
                'train/loss':     0.,
                'train/ins':      0.,
                'train/ce':       0.,
                'train/distill':  0.,
                'train/moe':      0.,
            }

            if len(self.buffer) > 0:

                with autocast():
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                    # sample enough new class samples
                    new_x = x.detach()
                    new_y = y.detach()
                    if batch_idx != 0:
                        buffer_cur_task = self.buffer_batch_size if task_id==0 else self.buffer_cur_task
                        cur_x, cur_y, _ = self.buffer.onlysample(buffer_cur_task, task=task_id)
                        if len(cur_x.shape) > 3:
                            new_x = torch.cat((x.detach(), cur_x))
                            new_y = torch.cat((y.detach(), cur_y))

                    if task_id > 0:
                        # balanced sampling for an ideal overall distribution
                        new_over_all = len(self.new_class_holder) / len(self.class_holder)
                        new_batch_size = min(
                            int(self.buffer_batch_size * new_over_all), x.size(0)
                        )
                        buffer_batch_size = min(
                            self.buffer_batch_size - new_batch_size,
                            self.buffer_per_class * len(self.class_holder)
                        )
                        mem_x, mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=task_id)
                        cat_x = torch.cat((x[:new_batch_size].detach(), mem_x))
                        cat_y = torch.cat((y[:new_batch_size].detach(), mem_y))

                        # rotate and augment
                        new_x = RandomFlip(new_x, 2)
                        new_y = new_y.repeat(2)
                        cat_x = RandomFlip(cat_x, 2)
                        cat_y = cat_y.repeat(2)

                        new_x = torch.cat((new_x, self.transform(new_x)))
                        new_y = torch.cat((new_y, new_y))
                        cat_x = torch.cat((cat_x, self.transform(cat_x)))
                        cat_y = torch.cat((cat_y, cat_y))

                        new_input_size = new_x.size(0)
                        cat_input_size = cat_x.size(0)

                        all_x = torch.cat((new_x, cat_x))
                        all_y = torch.cat((new_y, cat_y))
                        all_x = all_x.detach()
                        all_y = all_y.detach()

                        feat_list = self.model.features(all_x)
                        proj_list = self.model.head(feat_list, use_proj=True)
                        pred_list = self.model.head(feat_list, use_proj=False)
                        moe_pred  = self.model.moe(feat_list)

                        stu_feat = feat_list[self.expert]
                        stu_feat = self.model.final_addaption_layer(stu_feat)

                        # moe loss part
                        moe_new_pred = moe_pred[:new_input_size]
                        moe_cat_pred = moe_pred[new_input_size:]
                        moe_loss = 2 * F.cross_entropy(moe_cat_pred, cat_y)

                        moe_new_pred = moe_new_pred[:, self.new_class_holder]
                        new_y_onehot = F.one_hot(new_y, self.n_classes_num)
                        new_y_onehot = new_y_onehot[:, self.new_class_holder].float()
                        moe_loss += F.cross_entropy(moe_new_pred, new_y_onehot)

                        loss += moe_loss
                        loss_log['train/moe'] += moe_loss.item()

                        for i in range(len(feat_list)):
                            feat = feat_list[i]
                            proj = proj_list[i]
                            pred = pred_list[i]

                            new_feat = feat[:new_input_size]
                            cat_pred = feat[new_input_size:]
                            new_pred = pred[:new_input_size]
                            cat_pred = pred[new_input_size:]

                            # instance-wise contarstive loss
                            ins_loss = sup_con_loss(proj, self.ins_t, all_y)

                            # balanced cross entropy loss
                            ce_loss  = 2 * F.cross_entropy(cat_pred, cat_y)

                            new_pred = new_pred[:, self.new_class_holder]
                            new_y_onehot = F.one_hot(new_y, self.n_classes_num)
                            new_y_onehot = new_y_onehot[:, self.new_class_holder].float()
                            ce_loss += F.cross_entropy(new_pred, new_y_onehot)

                            # feature distillation loss
                            distill_loss = 0.
                            if i != self.expert:
                                distill_loss = torch.dist(
                                    F.normalize(stu_feat, dim=1), 
                                    F.normalize(feat.detach(), dim=1), p=2
                                )

                            loss += ins_loss + ce_loss + distill_loss
                            loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                            loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.
                            loss_log['train/distill'] += distill_loss.item() if distill_loss != 0. else 0.

                    else:
                        # rotate and augment
                        new_x = RandomFlip(new_x, 2)
                        new_y = new_y.repeat(2)

                        new_x = torch.cat((new_x, self.transform(new_x)))
                        new_y = torch.cat((new_y, new_y))
                        new_x = new_x.detach()
                        new_y = new_y.detach()

                        feat_list = self.model.features(new_x)
                        proj_list = self.model.head(feat_list, use_proj=True)
                        pred_list = self.model.head(feat_list, use_proj=False)
                        moe_pred  = self.model.moe(feat_list)

                        stu_feat = feat_list[self.expert]
                        stu_feat = self.model.final_addaption_layer(stu_feat)

                        # moe loss part
                        moe_loss = F.cross_entropy(moe_pred, new_y)
                        loss += moe_loss
                        loss_log['train/moe'] += moe_loss.item()

                        for i in range(len(feat_list)):
                            feat = feat_list[i]
                            proj = proj_list[i]
                            pred = pred_list[i]

                            # instance-wise contarstive loss
                            ins_loss = sup_con_loss(proj, self.ins_t, new_y)

                            # balanced cross entropy loss
                            ce_loss  = F.cross_entropy(pred, new_y)

                            # feature distillation loss
                            distill_loss = 0.
                            if i != self.expert:
                                distill_loss = torch.dist(
                                    F.normalize(stu_feat, dim=1), 
                                    F.normalize(feat.detach(), dim=1), p=2
                                )

                            loss += ins_loss + ce_loss + distill_loss
                            loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                            loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.
                            loss_log['train/distill'] += distill_loss.item() if distill_loss != 0. else 0.

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if epoch == 0:
                self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

            loss_log['train/loss'] = loss.item() if loss != 0. else 0.
            epoch_log_holder.append(loss_log)
            self.total_step += 1

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(f"==>>> it: {batch_idx}, loss: moe {loss_log['train/moe']:.3f} + ins {loss_log['train/ins']:.2f} + ce {loss_log['train/ce']:.3f} + distill {loss_log['train/distill']:.3f} = {loss:.6f}, {100 * (num_d / self.total_samples)}%")

        return epoch_log_holder

    def train(self, task_id, train_loader):
        self.model.train()
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log_holder = self.train_any_task(task_id, train_loader, epoch)
            train_log_holder.extend(epoch_log_holder)
            # self.buffer.print_per_task_num()
        return train_log_holder

    def test(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        if self.use_ncm:
            # calculate the class means for each feature layer
            print("\nCalculate class means for each layer...\n")
            self.class_means_ls = [{} for _ in range(4)]
            class_inputs = {cls: [] for cls in self.class_holder}
            for x, y in zip(self.buffer.x, self.buffer.y_int):
                class_inputs[y.item()].append(x)

            for cls, inputs in class_inputs.items():
                features = [[] for _ in range(4)]
                for ex in inputs:
                    return_features_ls = self.model.features(ex.unsqueeze(0))
                    for feat_id in range(4):
                        feature = return_features_ls[feat_id].detach().clone()
                        feature = F.normalize(feature, dim=1)
                        features[feat_id].append(feature.squeeze())

                for feat_id in range(4):
                    if len(features[feat_id]) == 0:
                        mu_y = torch.normal(
                            0, 1, size=tuple(self.model.features(x.unsqueeze(0))[feat_id].detach().size())
                        )
                        mu_y = mu_y.to(x.device)
                    else:
                        features[feat_id] = torch.stack(features[feat_id])
                        mu_y = features[feat_id].mean(0)

                    mu_y = F.normalize(mu_y.reshape(1, -1), dim=1)
                    self.class_means_ls[feat_id][cls] = mu_y.squeeze()

        all_acc_list = {'step': self.total_step}
        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_model(task_loader[j]['test'], j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_mean(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test moe classifier
        print(f"{'*'*100}\nTest with the moe output:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_moe(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['moe'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # # clear the calculated class_means
        # self.class_means_ls = None

        return acc_list, all_acc_list

    def test_buffer(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_buffer_task(j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_buffer_task_mean(j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def test_buffer_task(self, i, feat_id):
        # test specific layer's output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        x_i, y_i, _ = self.buffer.onlysample(self.buffer.current_index, task=i)

        if self.use_ncm:
            class_means = self.class_means_ls[feat_id]
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                features = self.model.features(x)[feat_id]
                features = F.normalize(features, dim=1)
                features = features.unsqueeze(2)
                means = torch.stack([class_means[cls] for cls in self.class_holder])
                means = torch.stack([means] * x.size(0))
                means = means.transpose(1, 2)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1).squeeze(1)
                pred = dists.min(1)[1]
                pred = torch.Tensor(self.class_holder)[pred].to(x.device)

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        else:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                pred = self.model(x)[feat_id]
                pred = pred.data.max(1, keepdim=True)[1]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_buffer_task_mean(self, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        x_i, y_i, _ = self.buffer.onlysample(self.buffer.current_index, task=i)

        if self.use_ncm:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                features_ls = self.model.features(x)
                dists_ls = []

                for feat_id in range(4):
                    class_means = self.class_means_ls[feat_id]
                    features = features_ls[feat_id]
                    features = F.normalize(features, dim=1)
                    features = features.unsqueeze(2)
                    means = torch.stack([class_means[cls] for cls in self.class_holder])
                    means = torch.stack([means] * x.size(0))
                    means = means.transpose(1, 2)
                    features = features.expand_as(means)
                    dists = (features - means).pow(2).sum(1).squeeze(1)
                    dists_ls.append(dists)

                dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
                dists = dists_ls.mean(dim=1).squeeze(1)
                pred = dists.min(1)[1]
                pred = torch.Tensor(self.class_holder)[pred].to(x.device)

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        else:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                pred = self.model(x)
                pred = torch.stack(pred, dim=1)
                pred = pred.mean(dim=1).squeeze(1)
                pred = pred.data.max(1, keepdim=True)[1]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_train(self, i, task_loader, feat_ids=[0,1,2,3]):
        # train accuracy of current task i
        self.model.eval()
        all_acc_list = {'step': self.total_step}

        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                acc = self.test_model(task_loader[i]['train'], i, feat_id=feat_id)
                acc_list[i] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            acc = self.test_model_mean(task_loader[i]['train'], i)
            acc_list[i] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def test_model(self, loader, i, feat_id):
        # test specific layer's output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            class_means = self.class_means_ls[feat_id]
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                features = self.model.features(data)[feat_id]
                features = F.normalize(features, dim=1)
                features = features.unsqueeze(2)
                means = torch.stack([class_means[cls] for cls in self.class_holder])
                means = torch.stack([means] * data.size(0))
                means = means.transpose(1, 2)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1).squeeze()
                pred = dists.min(1)[1]
                pred = torch.Tensor(self.class_holder)[pred].to(data.device)

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        else:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                pred = self.model(data)[feat_id]
                pred = pred.data.max(1, keepdim=True)[1]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_model_mean(self, loader, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()
                features_ls = self.model.features(data)
                dists_ls = []

                for feat_id in range(4):
                    class_means = self.class_means_ls[feat_id]
                    features = features_ls[feat_id]
                    features = F.normalize(features, dim=1)
                    features = features.unsqueeze(2)
                    means = torch.stack([class_means[cls] for cls in self.class_holder])
                    means = torch.stack([means] * data.size(0))
                    means = means.transpose(1, 2)
                    features = features.expand_as(means)
                    dists = (features - means).pow(2).sum(1).squeeze()
                    dists_ls.append(dists)

                dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
                dists = dists_ls.mean(dim=1).squeeze(1)
                pred = dists.min(1)[1]
                pred = torch.Tensor(self.class_holder)[pred].to(data.device)

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        else:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                pred = self.model(data)
                pred = torch.stack(pred, dim=1)
                pred = pred.mean(dim=1).squeeze()
                pred = pred.data.max(1, keepdim=True)[1]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_model_moe(self, loader, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()

            feat = self.model.features(data)
            pred = self.model.moe(feat)
            pred = pred.data.max(1, keepdim=True)[1]

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
