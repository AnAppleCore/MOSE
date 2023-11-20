import datetime
import os

import numpy as np
import torch
from torch.optim import Adam
from torchvision import datasets, transforms

from agent import get_agent
from models import get_model
from models.buffer import Buffer
from utils.util import Logger


def get_cifar_data_joint(dataset_name, batch_size, n_workers):

    size = [3, 32, 32]
    if dataset_name == "cifar10":
        class_num = 10
    elif dataset_name == "cifar100":
        class_num = 100

    dataset_path = './data/'
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    dataset = {}
    if dataset_name == "cifar10":
        dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    elif dataset_name == "cifar100":
        dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        
    Loader = {}

    Loader['train'] = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    Loader['test'] = torch.utils.data.DataLoader(
        dataset['test'],
        batch_size=64,
        shuffle=True,
        num_workers=n_workers,
    )

    return class_num, Loader, size


def multiple_run_joint(args):
    test_all_acc = torch.zeros(args.run_nums)
    last_test_all_acc = torch.zeros(args.run_nums)

    for run in range(args.run_nums):
        tmp_acc = []
        last_tmp_acc = []

        buffer_tmp_acc = []
        buffer_last_tmp_acc = []

        train_tmp_acc = []
        train_last_tmp_acc = []

        print('=' * 100)
        print(f"-----------------------------run {run} start--------------------------")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('=' * 100)
        class_num, task_loader, input_size = get_cifar_data_joint(
            dataset_name=args.dataset, batch_size=args.batch_size, n_workers=args.n_workers
        )
        args.n_classes = class_num

        setattr(args, 'run_name', f"{args.exp_name} run_{run:02d}")
        print(f"\nRun {run}: {args.run_name} {'*' * 50}\n")
        logger = Logger(args, base_dir=f"./outputs/{args.method}/{args.dataset}")
        buffer = Buffer(args, input_size).cuda()
        model = get_model(method_name=args.method, nclasses=class_num).cuda()
        optimizer = Adam(model.parameters(), args.lr, weight_decay=args.wd)
        agent = get_agent(
            method_name=args.method, model=model, 
            buffer=buffer, optimizer=optimizer, input_size=input_size, args=args
        )

        print(f"number of classifier parameters:\t {model.n_params/1e6:.2f}M", )
        print(f"buffer parameters (image size prod):\t {np.prod(buffer.bx.size())/1e6:.2f}M", )

        for i in range(1):
            print(f"\n-----------------------------run {run} task id:{i} start training-----------------------------")

            train_log_holder = agent.train(i, task_loader['train'])
            acc_list, all_acc_list = agent.test(i, task_loader)

            # empirical analysis of overfitting-underfitting dilemma
            buffer_acc_list, buffer_all_acc_list = agent.test_buffer(i, task_loader)
            train_acc_list, train_all_acc_list = agent.test_train(i, task_loader)

            tmp_acc.append(acc_list)
            last_tmp_acc.append(all_acc_list['3'])

            buffer_tmp_acc.append(buffer_acc_list)
            buffer_last_tmp_acc.append(buffer_all_acc_list['3'])

            train_tmp_acc.append(train_acc_list)
            train_last_tmp_acc.append(train_all_acc_list['3'])


            logger.log_losses(train_log_holder)
            logger.log_accs(all_acc_list)

            # record the intermediate final accs
            for feat_id, acc_list_id in all_acc_list.items():
                test_accuracy_id = acc_list_id[:i+1].mean()
                logger.log_scalars({
                    f"test/{feat_id}_avg_acc":       test_accuracy_id,
                }, step=agent.total_step)

        test_accuracy = acc_list.mean()
        test_all_acc[run] = test_accuracy
        
        tmp_acc = np.array(tmp_acc)

        logger.log_scalars({
            'test/final_avg_acc':       test_accuracy,
            'metrics/buffer_n_bits':        agent.buffer.n_bits / 1e6,
            'metrics/model_n_params':       agent.model.n_params / 1e6
        }, step=agent.total_step+1, verbose=True)

        logger.log_accs_table(
            name='task_accs_table', accs_list=tmp_acc,
            step=agent.total_step+1, verbose=True
        )

        # record the last scalars
        last_acc_list = all_acc_list['3']
        last_test_accuracy = last_acc_list.mean()
        last_test_all_acc[run] = last_test_accuracy

        last_tmp_acc = np.array(last_tmp_acc)

        logger.log_scalars({
            'test/last_final_avg_acc':       last_test_accuracy,
        }, step=agent.total_step+1, verbose=True)

        logger.log_accs_table(
            name='last_task_accs_table', accs_list=last_tmp_acc,
            step=agent.total_step+1, verbose=True
        )


        buffer_tmp_acc = np.array(buffer_tmp_acc)
        buffer_last_tmp_acc = np.array(buffer_last_tmp_acc)

        train_tmp_acc = np.array(train_tmp_acc)
        train_last_tmp_acc = np.array(train_last_tmp_acc)

        logger.log_accs_table(
            name='buffer_task_accs_table', accs_list=buffer_tmp_acc,
            step=agent.total_step+1, verbose=True
        )
        logger.log_accs_table(
            name='buffer_last_task_accs_table', accs_list=buffer_last_tmp_acc,
            step=agent.total_step+1, verbose=True
        )

        logger.log_accs_table(
            name='train_task_accs_table', accs_list=train_tmp_acc,
            step=agent.total_step+1, verbose=True
        )
        logger.log_accs_table(
            name='train_last_task_accs_table', accs_list=train_last_tmp_acc,
            step=agent.total_step+1, verbose=True
        )

        print('=' * 100)
        print("{}th run's Test result: Accuracy: {:.2f}%".format(run, test_accuracy))
        print('=' * 100)

        logger.close()

    print(f"\n{'=' * 100}")
    print(f"total {args.run_nums}runs last test acc results: {last_test_all_acc}")

    print(f"\n{'=' * 100}")
    print(f"total {args.run_nums}runs test acc results: {test_all_acc}")