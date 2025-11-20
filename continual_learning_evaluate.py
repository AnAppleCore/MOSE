import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from agent import get_agent
from experiment.dataset import get_data
from models import get_model
from models.buffer import Buffer
from utils.util import compute_performance


def estimate_flops_per_sample(model, input_size, device):
    """Approximate FLOPs for a single forward pass on one sample.

    We count only Conv2d and Linear layers (MACs * 2). This is an
    approximation of inference cost per sample, not including backward.
    """
    model_eval = model.to(device).eval()
    flops = 0.0

    def conv_hook(module, input, output):
        nonlocal flops
        if not input:
            return
        x = input[0]
        out = output
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out is None or x is None:
            return
        if out.dim() != 4:
            return
        batch_size = out.shape[0]
        Cin = module.in_channels
        Cout = module.out_channels
        Kh, Kw = module.kernel_size
        groups = module.groups
        H_out, W_out = out.shape[2], out.shape[3]
        macs_per_out = (Cin / groups) * Kh * Kw
        total_macs = batch_size * Cout * H_out * W_out * macs_per_out
        flops += 2.0 * total_macs

    def linear_hook(module, input, output):
        nonlocal flops
        if not input:
            return
        x = input[0]
        if x is None or x.dim() == 0:
            return
        batch_size = x.shape[0]
        in_features = module.in_features
        out_features = module.out_features
        flops += 2.0 * batch_size * in_features * out_features

    handles = []
    for m in model_eval.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    dummy = torch.randn(1, *input_size, device=device)
    with torch.no_grad():
        try:
            _ = model_eval(dummy)
        except Exception:
            # Some models may require extra args (e.g., use_proj)
            try:
                _ = model_eval(dummy, False)
            except Exception:
                for h in handles:
                    h.remove()
                return None

    for h in handles:
        h.remove()

    # dummy batch size = 1 -> flops is per-sample
    return float(flops)


def parse_train_time_minutes(log_path):
    """Parse total training wall time (minutes) from a text log.

    We look for timestamps in format YYYY-MM-DD HH:MM:SS and take the
    difference between the earliest and latest occurrence.
    """
    if log_path is None or not os.path.isfile(log_path):
        return None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return None

    matches = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
    if len(matches) < 2:
        return None

    try:
        times = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in matches]
    except ValueError:
        return None

    delta = max(times) - min(times)
    return delta.total_seconds() / 60.0


def load_train_args(exp_dir: str) -> argparse.Namespace:
    """Load the original training arguments from params.json in exp_dir."""
    params_path = os.path.join(exp_dir, "params.json")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"params.json not found in {exp_dir}")

    with open(params_path, "r") as f:
        params = json.load(f)
    # Convert dict to argparse.Namespace for compatibility with training code
    return argparse.Namespace(**params)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a continual-learning checkpoint and evaluate its performance."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Directory of a single run (contains params.json and final.pt).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional path to checkpoint; default is <exp_dir>/final.pt.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Override GPU id used for evaluation.",
    )
    parser.add_argument(
        "--train_log",
        type=str,
        default=None,
        help=(
            "Optional path to the training log (stdout). Used to estimate total "
            "training time in minutes."
        ),
    )
    args = parser.parse_args()

    # Recreate the training args from params.json
    train_args = load_train_args(args.exp_dir)

    if args.gpu_id is not None:
        train_args.gpu_id = args.gpu_id

    device = torch.device(
        f"cuda:{train_args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(train_args.gpu_id)

    # Recreate data, model, buffer and agent exactly as in training
    data, class_num, class_per_task, task_loader, input_size = get_data(
        dataset_name=train_args.dataset,
        batch_size=train_args.batch_size,
        n_workers=train_args.n_workers,
        n_tasks=train_args.n_tasks,
    )
    train_args.n_classes = class_num

    buffer = Buffer(train_args, input_size).to(device)
    model = get_model(method_name=train_args.method, nclasses=class_num).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), train_args.lr, weight_decay=train_args.wd
    )
    agent = get_agent(
        method_name=train_args.method,
        model=model,
        buffer=buffer,
        optimizer=optimizer,
        input_size=input_size,
        args=train_args,
    )

    # Estimate FLOPs per sample (forward pass) and training time (from log)
    flops_per_sample = estimate_flops_per_sample(model, input_size, device)
    train_minutes = parse_train_time_minutes(args.train_log) if args.train_log else None

    print("=" * 100)
    print("Approximate compute cost (forward pass, Conv2d + Linear only):")
    if flops_per_sample is not None:
        print(f"FLOPs per sample: {flops_per_sample:.2e}")
        print(f"MFLOPs per sample: {flops_per_sample / 1e6:.2f}")
        print(f"GFLOPs per sample: {flops_per_sample / 1e9:.3f}")
    else:
        print("FLOPs estimation not available for this model.")

    if train_minutes is not None:
        print(
            f"Estimated total training time from log ({args.train_log}): "
            f"{train_minutes:.2f} min"
        )
    elif args.train_log is not None:
        print(f"Could not parse training time from log: {args.train_log}")
    else:
        print("No training log provided; skip training time estimation.")

    # Load checkpoint (model + buffer)
    ckpt_path = args.ckpt_path or os.path.join(args.exp_dir, "final.pt")
    print(f"Loading checkpoint from: {ckpt_path}")
    agent.load_checkpoint(ckpt_path)

    # For MOSE (and similar), rebuild class_holder from the buffer if needed
    if hasattr(agent, "class_holder") and hasattr(agent, "buffer"):
        y_int = getattr(agent.buffer, "y_int", None)
        if y_int is not None and len(y_int) > 0:
            try:
                unique_labels = torch.unique(y_int).cpu().tolist()
                agent.class_holder = sorted(unique_labels)
            except Exception:
                pass

    # Evaluate final model on all tasks (using the same test routine as during training)
    num_tasks = len(task_loader)
    last_task_id = num_tasks - 1

    print("=" * 100)
    print(f"Evaluating final model on {num_tasks} tasks...")
    acc_list, all_acc_list = agent.test(last_task_id, task_loader)

    expert_key = str(getattr(train_args, "expert", "3"))
    if expert_key in all_acc_list:
        final_accs = all_acc_list[expert_key]
    else:
        final_accs = acc_list

    final_accs = np.array(final_accs[:num_tasks], dtype=float)
    final_avg_acc = final_accs.mean()

    print(f"Final per-task accuracies (expert {expert_key}): {final_accs}")
    print(f"Final average accuracy over {num_tasks} tasks: {final_avg_acc:.2f}%")

    # If task_accs_table.csv exists, also compute standard continual-learning metrics
    csv_path = os.path.join(args.exp_dir, "task_accs_table.csv")
    if os.path.isfile(csv_path):
        print("=" * 100)
        print(f"Found task_accs_table.csv at: {csv_path}")
        table = pd.read_csv(csv_path, index_col=0)
        acc_mat = table.values.astype(float)
        # Shape expected by compute_performance: (n_run, n_tasks, n_tasks)
        end_task_acc_arr = acc_mat[None, :, :]

        avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(
            end_task_acc_arr
        )

        print("Continual learning metrics from logged accuracies:")
        print(
            f"Avg_End_Acc: {avg_end_acc[0]:.2f} ± {avg_end_acc[1]:.2f}"
        )
        print(
            f"Avg_End_Fgt: {avg_end_fgt[0]:.2f} ± {avg_end_fgt[1]:.2f}"
        )
        print(f"Avg_Acc:     {avg_acc[0]:.2f} ± {avg_acc[1]:.2f}")
        print(f"Avg_Bwt+:    {avg_bwtp[0]:.2f} ± {avg_bwtp[1]:.2f}")
        print(f"Avg_Fwt:     {avg_fwt[0]:.2f} ± {avg_fwt[1]:.2f}")
    else:
        print("No task_accs_table.csv found; only final accuracies are reported.")


if __name__ == "__main__":
    main()

