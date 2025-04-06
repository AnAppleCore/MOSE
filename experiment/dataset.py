import os
from functools import partial

import numpy as np
import torch
from kornia.augmentation.auto import RandAugment, TrivialAugment
from torchvision import datasets, transforms

from experiment.tinyimagenet import MyTinyImagenet


def get_mnist_data(dataset_name, batch_size, n_workers, **kwargs):
    data = {}
    size = [3, 32, 32]
    task_num = 5
    class_num = 10
    data_dir = './data/binary_mnist_5/'
    class_per_task = class_num // task_num

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dataset_path = './data/'
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean * 3, std * 3)
        ])

        dataset = {
            'train': datasets.MNIST(dataset_path, train=True, download=True, transform=transform),
            'test': datasets.MNIST(dataset_path, train=False, download=True, transform=transform)
        }

        for task_id in range(task_num):
            data[task_id] = {}
            for data_type in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
                data[task_id][data_type] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(task_id * class_per_task, (task_id + 1) * class_per_task):
                        data[task_id][data_type]['x'].append(image)
                        data[task_id][data_type]['y'].append(label)

        for task_id in data:
            for data_type in ['train', 'test']:
                data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, *size)
                data[task_id][data_type]['y'] = torch.LongTensor(np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
                torch.save(data[task_id][data_type]['x'], 
                           os.path.join(os.path.expanduser(data_dir), f'data{task_id}{data_type}x.bin'))
                torch.save(data[task_id][data_type]['y'], 
                           os.path.join(os.path.expanduser(data_dir), f'data{task_id}{data_type}y.bin'))

    data = {}
    ids = list(range(task_num))
    print('Task order =', ids)
    for task_id in ids:
        data[task_id] = dict.fromkeys(['train', 'test'])
        for data_type in ['train', 'test']:
            data[task_id][data_type] = {'x': [], 'y': []}
            data[task_id][data_type]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), f'data{task_id}{data_type}x.bin'))
            data[task_id][data_type]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), f'data{task_id}{data_type}y.bin'))

    Loader = {}
    for task_id in range(task_num):
        Loader[task_id] = dict.fromkeys(['train', 'test'])
        for data_type in ['train', 'test']:
            dataset = torch.utils.data.TensorDataset(
                data[task_id][data_type]['x'], data[task_id][data_type]['y'])
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size if data_type == 'train' else 64,
                shuffle=True,
                num_workers=n_workers
            )
            Loader[task_id][data_type] = loader

    print("MNIST data and loader prepared")
    return data, class_num, class_per_task, Loader, size


def get_cifar_data(dataset_name, batch_size, n_workers, **kwargs):
    data = {}
    size = [3, 32, 32]
    if dataset_name == "cifar10":
        task_num = 5
        class_num = 10
        data_dir = './data/binary_cifar10_5/'
    elif dataset_name == "cifar100":
        task_num = 10
        class_num = 100
        data_dir = './data/binary_cifar100_10/'
    class_per_task = class_num // task_num

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dataset_path = './data/CIFAR'
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = {}
        if dataset_name == "cifar10":
            dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
                # [transforms.ToTensor(), RandAugment(n=2, m=10), transforms.Normalize(mean, std)]))
                # [transforms.ToTensor(), TrivialAugment(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        elif dataset_name == "cifar100" or dataset_name == "cifar100_50":
            dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for task_id in range(task_num):
            data[task_id] = {}
            for data_type in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
                data[task_id][data_type] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(class_per_task * task_id, class_per_task * (task_id + 1)):
                        data[task_id][data_type]['x'].append(image)
                        data[task_id][data_type]['y'].append(label)

        # save
        for task_id in data.keys():
            for data_type in ['train', 'test']:
                data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, size[0], size[1], size[2])
                data[task_id][data_type]['y'] = torch.LongTensor(np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
                torch.save(data[task_id][data_type]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'x.bin'))
                torch.save(data[task_id][data_type]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])

        dataset_new_train = torch.utils.data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = torch.utils.data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size


def get_tinyimagenet(batch_size, n_workers, n_tasks=100):
    data = {}
    size = [3, 64, 64]
    task_num = n_tasks
    class_num = 200
    class_per_task = class_num // task_num

    base_path = './data/TINYIMG'
    data_dir = f'./data/binary_tiny200_{task_num}'

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dat = {}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyTinyImagenet(base_path, train=True, download=True, transform=test_transform)
        test = MyTinyImagenet(base_path, train=False, download=True, transform=test_transform)

        dat['train'] = train
        dat['test'] = test
        for t in range(task_num):
            data[t] = {}
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(class_per_task * t, class_per_task * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)

        # and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])

        dataset_new_train = torch.utils.data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = torch.utils.data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size


DATASETS = {
    'cifar10':  partial(get_cifar_data, dataset_name='cifar10'),
    'cifar100': partial(get_cifar_data, dataset_name='cifar100'),
    'tiny_imagenet': get_tinyimagenet,
    'mnist': partial(get_mnist_data, dataset_name='mnist')
}


def get_data(dataset_name, *args, **kwargs):
    if dataset_name in DATASETS.keys():
        return DATASETS[dataset_name](*args, **kwargs)
    else:
        raise Exception('unknown dataset!')