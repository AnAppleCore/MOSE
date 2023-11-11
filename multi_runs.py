import datetime

import numpy as np
import torch
from torch.optim import Adam

from agent import get_agent
from experiment.dataset import get_data
from models import get_model
from models.buffer import Buffer
from utils.util import Logger, compute_performance


def multiple_run(args):
    test_all_acc = torch.zeros(args.run_nums)
    last_test_all_acc = torch.zeros(args.run_nums)

    accuracy_list = []
    last_accuracy_list = []
    for run in range(args.run_nums):
        tmp_acc = []
        last_tmp_acc = []
        print('=' * 100)
        print(f"-----------------------------run {run} start--------------------------")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('=' * 100)
        data, class_num, class_per_task, task_loader, input_size = get_data(
            dataset_name=args.dataset, batch_size=args.batch_size, n_workers=args.n_workers
        )
        args.n_classes = class_num

        setattr(args, 'run_name', f"{args.exp_name} run_{run:02d}")
        print(f"\nRun {run}: {args.run_name} {'*' * 50}\n")
        logger = Logger(args, base_dir=f"./outputs/{args.method}/{args.dataset}")
        
        buffer = Buffer(args, input_size).cuda()
        model = get_model(method_name=args.method, nclasses=class_num).cuda()
        optimizer = Adam(model.parameters(), args.lr, weight_decay=1e-4)
        agent = get_agent(
            method_name=args.method, model=model, 
            buffer=buffer, optimizer=optimizer, input_size=input_size, args=args
        )

        print(f"number of classifier parameters:\t {model.n_params/1e6:.2f}M", )
        print(f"buffer parameters (image size prod):\t {np.prod(buffer.bx.size())/1e6:.2f}M", )

        for i in range(len(task_loader)):
            print(f"\n-----------------------------run {run} task id:{i} start training-----------------------------")

            train_log_holder = agent.train(i, task_loader[i]['train'])
            acc_list, all_acc_list = agent.test(i, task_loader)
            tmp_acc.append(acc_list)
            last_tmp_acc.append(all_acc_list['3'])

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

        logger.log_accs_table(
            name='task_accs_table', accs_list=tmp_acc,
            step=agent.total_step+1, verbose=True
        )

        # record the last scalars
        last_acc_list = all_acc_list['3']
        last_test_accuracy = last_acc_list.mean()
        last_test_all_acc[run] = last_test_accuracy
        last_tmp_acc = np.array(last_tmp_acc)
        last_avg_fgt = (last_tmp_acc.max(0) - last_tmp_acc[-1, :]).mean()
        last_avg_bwt = (last_tmp_acc[-1, :] - np.diagonal(last_tmp_acc)).mean()
        last_accuracy_list.append(last_tmp_acc)

        logger.log_scalars({
            'test/last_final_avg_acc':       last_test_accuracy,
            'test/last_final_avg_fgt':       last_avg_fgt,
            'test/last_final_avg_bwt':       last_avg_bwt,
        }, step=agent.total_step+1, verbose=True)

        logger.log_accs_table(
            name='last_task_accs_table', accs_list=last_tmp_acc,
            step=agent.total_step+1, verbose=True
        )

        print('=' * 100)
        print("{}th run's Test result: Accuracy: {:.2f}%".format(run, test_accuracy))
        print('=' * 100)

        logger.close()

    last_accuracy_array = np.array(last_accuracy_list)
    last_avg_end_acc, last_avg_end_fgt, last_avg_acc, last_avg_bwtp, last_avg_fwt = compute_performance(last_accuracy_array)
    print(f"\n{'=' * 100}")
    print(f"total {args.run_nums}runs last test acc results: {last_test_all_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(last_avg_end_acc, last_avg_end_fgt, last_avg_acc, last_avg_bwtp, last_avg_fwt))
    print('=' * 100)

    accuracy_array = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
    print(f"\n{'=' * 100}")
    print(f"total {args.run_nums}runs test acc results: {test_all_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    print('=' * 100)
