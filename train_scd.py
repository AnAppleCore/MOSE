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
    def __init__(self, model, buffer, optimizer, n_classes_num, class_per_task, input_size, args, fea_dim=128):
        self.model = model
        self.optimizer = optimizer
        self.oop_base = n_classes_num
        self.oop = args.oop
        self.n_classes_num = n_classes_num
        self.fea_dim = fea_dim
        self.classes_mean = torch.zeros((n_classes_num, fea_dim), requires_grad=False).cuda()
        self.class_per_task = class_per_task
        self.class_holder = []
        self.mixup_base_rate = args.mixup_base_rate
        self.ins_t = args.ins_t # 0.07
        self.proto_t = args.proto_t
        self.img_size = input_size

        self.buffer = buffer
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_per_class = 7

        self.OPELoss = OPELoss(self.class_per_task, temperature=self.proto_t)

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

        self.APF = AdaptivePrototypicalFeedback(self.buffer, args.mixup_base_rate, args.mixup_p, args.mixup_lower, args.mixup_upper,
                                  args.mixup_alpha, self.class_per_task)

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
        # log_softmax_outputs = F.log_softmax(outputs/temp, dim=1)
        # softmax_targets = F.softmax(targets/temp, dim=1)
        # return loss_func(log_softmax_outputs, softmax_targets)
        outputs = F.normalize(outputs)
        targets = F.normalize(targets)
        return loss_func(outputs, targets)

    def train_any_task(self, task_id, train_loader):
        num_d = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach())

            loss = 0.

            loss_log = {
                'ins': 0.,
                'ce': 0.,
                'distill': 0.,
            }

            if len(self.buffer) > 0:

                with autocast():
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
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

                        ins_loss = 0.
                        ins_loss = sup_con_loss(proj, 0.07, all_y)
                        # ce_loss = 0.
                        ce_loss  = F.cross_entropy(pred, all_y)

                        distill_loss = 0.
                        if i != len(feat_list)-1:
                            distill_loss = self.CrossEntropyDistill(pred, pred_list[i+1].detach(), 3.0)

                        loss += ins_loss + ce_loss + distill_loss
                        loss_log['ins'] += ins_loss
                        loss_log['ce'] += ce_loss
                        loss_log['distill'] += distill_loss

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
            self.buffer.print_per_task_num()

    def test(self, i, task_loader):
        self.model.eval()
        for feat_id in [3]:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            self.class_means = {}
            self.seen_classes = list(set(self.buffer.y_int.tolist()))
            class_inputs = {cls: [] for cls in self.seen_classes}
            for x, y in zip(self.buffer.x, self.buffer.y_int):
                class_inputs[y.item()].append(x)

            for cls, inputs in class_inputs.items():
                features = []
                for ex in inputs:
                    # feature = self.model.final_feature(ex.unsqueeze(0)).detach().clone()
                    feature = self.model.features(ex.unsqueeze(0))[feat_id].detach().clone()
                    feature = F.normalize(feature, dim=1)
                    features.append(feature.squeeze())

                if len(features) == 0:
                    mu_y = torch.normal(
                        # 0, 1, size=tuple(self.model.final_feature(x.unsqueeze(0)).detach().size())
                        0, 1, size=tuple(self.model.features(x.unsqueeze(0))[feat_id].detach().size())
                    )
                    mu_y = mu_y.to(x.device)
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0)
                mu_y = F.normalize(mu_y.reshape(1, -1), dim=1)
                self.class_means[cls] = mu_y.squeeze()

            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_model(task_loader[j]['test'], j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list

    '''
    def test(self, i, task_loader):
        self.model.eval()
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list
    '''

    def test_model(self, loader, i, feat_id):
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()

            # features = self.model.final_feature(data)
            features = self.model.features(data)[feat_id]
            features = F.normalize(features, dim=1)
            features = features.unsqueeze(2)
            means = torch.stack([self.class_means[cls] for cls in self.seen_classes])
            means = torch.stack([means] * data.size(0))
            means = means.transpose(1, 2)
            features = features.expand_as(means)
            dists = (features - means).pow(2).sum(1).squeeze()
            pred = dists.min(1)[1]
            pred = torch.Tensor(self.seen_classes)[pred].to(data.device)

            num += data.size()[0]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    '''
    def test_model(self, loader, i):
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            pred = self.model(data)[-1]
            Pred = pred.data.max(1, keepdim=True)[1]
            num += data.size()[0]
            correct += Pred.eq(target.data.view_as(Pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy
    '''