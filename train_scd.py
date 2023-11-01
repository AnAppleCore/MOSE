import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from utils.rotation_transform import RandomFlip
from utils import my_transform as TL
from losses.loss import Supervised_NT_xent_n, Supervised_NT_xent_uni, sup_con_loss, Supervised_NT_xent_distill
from copy import deepcopy
from modules.OPE import OPELoss
from modules.APF import AdaptivePrototypicalFeedback
import kornia
from info_nce import InfoNCE
from utils.random_transorm import RandAugment
from torchvision import transforms


class TrainLearner_SCD(object):
    def __init__(self, model:nn.Module, buffer, optimizer, n_classes_num, class_per_task, input_size, args, fea_dim=128):
        self.model = model
        self.optimizer = optimizer
        self.oop_base = n_classes_num
        self.n_classes_num = n_classes_num
        self.fea_dim = fea_dim
        self.classes_mean = torch.zeros((n_classes_num, fea_dim), requires_grad=False).cuda()
        self.class_per_task = class_per_task
        self.class_holder = []
        self.ins_t = args.ins_t
        self.img_size = input_size

        self.buffer = buffer
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_per_class = 7
        self.buffer_cur_task = (self.buffer_batch_size // 2) - args.batch_size

        self.dataset = args.dataset
        if args.dataset == "cifar10":
            self.sim_lambda = 0.5
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.sim_lambda = 1.0
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.sim_lambda = 1.0
            self.total_samples = 10000
        self.print_num = self.total_samples // 10

        hflip = TL.HorizontalFlipLayer().cuda()
        with torch.no_grad():
            resize_scale = (0.3, 1.0)
            color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
            resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[input_size[1], input_size[2], input_size[0]]).cuda()
            self.transform = torch.nn.Sequential(
                hflip,
                color_gray,
                resize_crop)

        self.scaler = GradScaler()

        self.scr_transform = nn.Sequential(
            kornia.augmentation.RandomResizedCrop(
                size=(input_size[1], input_size[2]), scale=(0.2, 1.)
            ),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.2)
        )

        self.to_tensor = transforms.Compose([transforms.ToTensor(),])
        self.rand_ops = RandAugment(1, 5)
        self.rand_transform = self.to_tensor
        self.rand_transform.transforms.insert(0, self.rand_ops)

    def randaug(self, imgs:torch.Tensor):
        B = imgs.size(0)
        device = imgs.device
        imgs = [transforms.ToPILImage()(img) for img in imgs]
        aug_imgs = [self.rand_transform(img).reshape([1,*self.img_size]) for img in imgs]
        aug_imgs = torch.cat((aug_imgs), dim=0).to(device)
        return aug_imgs

    def CrossEntropyDistill(self, outputs, targets, temp=3.0):
        # targets = F.one_hot(targets, self.n_classes_num)
        log_softmax_outputs = F.log_softmax(outputs/temp, dim=1)
        softmax_targets = F.softmax(targets/temp, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
    
    def infonce_distill(self, outputs, targets, temp=3.0):
        loss_func = InfoNCE(temp)
        return loss_func(outputs, targets)
    
    def kl_distill(self, outputs, targets, temp=1.0):
        loss_func = nn.KLDivLoss(reduction='mean')
        outputs = F.normalize(outputs)
        targets = F.normalize(targets)
        return loss_func(outputs, targets)

    def train_any_task(self, task_id, train_loader):
        num_d = 0
        new_class_holder = []
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach().item())
                    new_class_holder.append(Y[j].detach().item())

            loss = 0.

            loss_log = {
                'ins': 0.,
                'ce': 0.,
                'distill': 0.,
            }

            if len(self.buffer) > 0:

                with autocast():
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                    # sample enough new class samples
                    if batch_idx != 0:
                        buffer_cur_task = self.buffer_batch_size if task_id==0 else self.buffer_cur_task
                        cur_x, cur_y, _ = self.buffer.onlysample(buffer_cur_task, task=task_id)
                        if len(cur_x.shape) > 3:
                            new_x = torch.cat((x.detach(), cur_x))
                            new_y = torch.cat((y.detach(), cur_y))
                        else:
                            new_x = x.detach()
                            new_y = y.detach()
                    else:
                        new_x = x.detach()
                        new_y = y.detach()
                    new_x.requires_grad_()

                    if task_id > 0:
                        # balanced sampling for an ideal overall distribution
                        new_over_all = len(new_class_holder) / len(self.class_holder)
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
                        cat_x.requires_grad_()

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

                        feat_list = self.model.features(all_x)
                        proj_list = self.model.head(feat_list, use_proj=True)
                        pred_list = self.model.head(feat_list, use_proj=False)

                        for i in range(len(feat_list)):
                            feat = feat_list[i]
                            proj = proj_list[i]
                            pred = pred_list[i]

                            new_pred = pred[:new_input_size]
                            cat_pred = pred[new_input_size:]

                            ins_loss = 0.
                            ins_loss = sup_con_loss(proj, 0.07, all_y)

                            ce_loss = 0.
                            # balanced cross entropy loss
                            ce_loss += 2 * F.cross_entropy(cat_pred, cat_y)

                            # new cross entropy loss
                            new_pred = new_pred[:, new_class_holder]
                            new_y_onehot = F.one_hot(new_y, self.n_classes_num)
                            new_y_onehot = new_y_onehot[:, new_class_holder].float()
                            ce_loss += F.cross_entropy(new_pred, new_y_onehot)

                            distill_loss = 0.
                            if i != len(feat_list)-1:
                                distill_loss = self.CrossEntropyDistill(pred, pred_list[i+1].detach(), 3.0)

                            loss += ins_loss + ce_loss + distill_loss
                            loss_log['ins'] += ins_loss
                            loss_log['ce'] += ce_loss
                            loss_log['distill'] += distill_loss
                    else:
                        # rotate and augment
                        new_x = RandomFlip(new_x, 2)
                        new_y = new_y.repeat(2)

                        new_x = torch.cat((new_x, self.transform(new_x)))
                        new_y = torch.cat((new_y, new_y))

                        feat_list = self.model.features(new_x)
                        proj_list = self.model.head(feat_list, use_proj=True)
                        pred_list = self.model.head(feat_list, use_proj=False)

                        for i in range(len(feat_list)):
                            feat = feat_list[i]
                            proj = proj_list[i]
                            pred = pred_list[i]

                            ins_loss = 0.
                            ins_loss = sup_con_loss(proj, 0.07, new_y)

                            ce_loss = 0.
                            ce_loss += F.cross_entropy(pred, new_y)

                            distill_loss = 0.
                            if i != len(feat_list)-1:
                                distill_loss = self.CrossEntropyDistill(pred, pred_list[i+1].detach(), 3.0)

                            loss += ins_loss + ce_loss + distill_loss
                            loss_log['ins'] += ins_loss
                            loss_log['ce'] += ce_loss
                            loss_log['distill'] += distill_loss

                    '''
                    x = x.requires_grad_()
                    buffer_batch_size = min(self.buffer_batch_size, self.buffer_per_class * len(self.class_holder))
                    mem_x, mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=None)

                    cat_x = torch.cat((x, mem_x))
                    cat_y = torch.cat((y, mem_y))

                    # rotate
                    cat_x = RandomFlip(cat_x, 2)
                    cat_y = cat_y.repeat(2)

                    cat_x_aug = self.transform(cat_x)
                    # cat_x_aug = self.scr_transform(cat_x)
                    all_x = torch.cat((cat_x, cat_x_aug))
                    all_y = torch.cat((cat_y, cat_y))

                    feat_list = self.model.features(all_x)
                    proj_list = self.model.head(feat_list, use_proj=True)
                    pred_list = self.model.head(feat_list, use_proj=False)

                    last_feat = feat_list[-1].detach()
                    last_proj = proj_list[-1].detach()
                    last_pred = pred_list[-1].detach()

                    for i in range(len(feat_list)):
                        feat = feat_list[i]
                        proj = proj_list[i]
                        pred = pred_list[i]

                        # ins_loss = 0.
                        ins_loss = sup_con_loss(proj, 0.07, all_y)
                        # ce_loss = 0.
                        ce_loss  = F.cross_entropy(pred, all_y)

                        distill_loss = 0.
                        # if i != len(feat_list)-1:
                        #     distill_loss = self.CrossEntropyDistill(pred, pred_list[i+1].detach(), 3.0)

                        loss += ins_loss + ce_loss + distill_loss
                        loss_log['ins'] += ins_loss
                        loss_log['ce'] += ce_loss
                        loss_log['distill'] += distill_loss
                    '''

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(f"==>>> it: {batch_idx}, loss: ins {loss_log['ins']:.2f} + ce {loss_log['ce']:.3f} + distill {loss_log['distill']:.3f} = {loss:.6f}, {100 * (num_d / self.total_samples)}%")

    def train(self, task_id, train_loader):
        self.model.train()
        for epoch in range(1):
            # if task_id == 0:
            #     self.train_task0(task_id, train_loader)
            # else:
            #     self.train_other_tasks(task_id, train_loader)
            self.train_any_task(task_id, train_loader)
            # self.buffer.print_per_task_num()


    def test(self, i, task_loader, feat_ids=[3]):

        # calculate the class means for each feature layer
        print("\nCalculate class means for each layer...\n")
        self.model.eval()
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

        # test with ncm classifier for each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_model(task_loader[j]['test'], j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test with ncm classifier with mean dists
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_mean(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list

    def test_model(self, loader, i, feat_id):
        # test specific layer's ncm output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
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

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_model_mean(self, loader, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
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

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy


    # line classifier here
    '''
    def test(self, i, task_loader, feat_ids=[3]):
        # linear classifier
        self.model.eval()
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_model(task_loader[j]['test'], j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        print(f"{'*'*100}\nTest with the mean output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_mean(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list

    def test_model(self, loader, i, feat_id):
        # test specific layer's ncm output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            pred = self.model(data)[feat_id]
            # pred = self.model(data)
            # pred = torch.stack(pred, dim=1)
            # pred = pred.mean(dim=1).squeeze()
            Pred = pred.data.max(1, keepdim=True)[1]
            num += data.size()[0]
            correct += Pred.eq(target.data.view_as(Pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy
    
    def test_model_mean(self, loader, i):
        # test with mean pred of all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            pred = self.model(data)
            pred = torch.stack(pred, dim=1)
            pred = pred.mean(dim=1).squeeze()
            Pred = pred.data.max(1, keepdim=True)[1]
            num += data.size()[0]
            correct += Pred.eq(target.data.view_as(Pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy
    '''

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
