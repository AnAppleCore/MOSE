import datetime

import numpy as np
import torch
from torch.optim import Adam
from experiment.dataset import get_data
from models.buffer import Buffer
from train_mose import TrainLearner_MOSE
from models.Resnet18_SD import resnet18_sd
from utils.util import Logger, compute_performance


def multiple_run(args):
    test_all_acc = torch.zeros(args.run_nums)

    accuracy_list = []
    for run in range(args.run_nums):
        tmp_acc = []
        print('=' * 100)
        print(f"-----------------------------run {run} start--------------------------")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('=' * 100)
        data, class_num, class_per_task, task_loader, input_size = get_data(args.dataset, args.batch_size, args.n_workers)
        args.n_classes = class_num
        buffer = Buffer(args, input_size).cuda()

        setattr(args, 'run_name', f"{args.exp_name} run_{run:02d}")
        print(f"\nRun {run}: {args.run_name} {'*' * 50}\n")
        logger = Logger(args)

        model = resnet18_sd(class_num).cuda()
        optimizer = Adam(model.parameters(), lr=args.lr,  weight_decay=1e-4)
        agent = TrainLearner_MOSE(model, buffer, optimizer, class_num, class_per_task, input_size, args)

        print(f"number of classifier parameters:\t {model.n_params/1e6:.2f}M", )
        print(f"buffer parameters (image size prod):\t {np.prod(buffer.bx.size())/1e6:.2f}M", )

        for i in range(len(task_loader)):
            print(f"\n-----------------------------run {run} task id:{i} start training-----------------------------")

            train_log_holder = agent.train(i, task_loader[i]['train'])
            acc_list, all_acc_list = agent.test(i, task_loader)
            tmp_acc.append(acc_list)

            logger.log_losses(train_log_holder)
            logger.log_accs(all_acc_list)

        test_accuracy = acc_list.mean()
        test_all_acc[run] = test_accuracy
        
        tmp_acc = np.array(tmp_acc)
        avg_fgt = (tmp_acc.max(0) - tmp_acc[-1, :]).mean()
        avg_bwt = (tmp_acc[-1, :] - np.diagonal(tmp_acc)).mean()
        accuracy_list.append(tmp_acc)

        logger.log_scalars({
            'test/final_avg_acc':       test_accuracy,
            'test/final_avg_fgt':       avg_fgt,
            'test/final_avg_bwt':       avg_bwt,
            'metrics/buffer_n_bits':        agent.buffer.n_bits / 1e6,
            'metrics/model_n_params':       agent.model.n_params / 1e6
        }, step=agent.total_step+1, verbose=True)

        print('=' * 100)
        print("{}th run's Test result: Accuracy: {:.2f}%".format(run, test_accuracy))
        print('=' * 100)

        logger.close()

    accuracy_array = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
    print('=' * 100)
    print(f"total {args.run_nums}runs test acc results: {test_all_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    print('=' * 100)
