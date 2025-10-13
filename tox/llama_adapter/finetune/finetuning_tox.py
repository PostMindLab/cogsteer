import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path

import models_llama_adapter
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
from engine_finetuning import train_one_epoch, val_one_epoch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import pandas as pd

from llama import Tokenizer
import re
import wandb
def clean_text(text):
    text = text.replace('\n', ' ')
    
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub('', text)
    
    ipv4_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    text = ipv4_pattern.sub('', text)
    
    ipv6_pattern = re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b')
    text = ipv6_pattern.sub('', text)
    
    mac_like_pattern = re.compile(r'\b(?:[0-9a-fA-F]{1,2}:){5}[0-9a-fA-F]{1,2}\b')
    text = mac_like_pattern.sub('', text)
    
    return text

class InstructionDataset(Dataset):
    def __init__(self, data_path, model_path, max_words=30, partition="train"):

        if partition == "train":
            dataset = pd.read_csv(data_path + 'train.csv')
        else:
            dataset = pd.read_csv(data_path + 'test.csv')

        self.max_words = max_words
        tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer1 = tokenizer
        self.ann = dataset["comment_text"].tolist()

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        ann = clean_text(ann)
        
        example = torch.tensor(self.tokenizer1.encode(ann, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = example.clone()
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return example, labels, example_mask


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--model", default="llama7B_adapter", type=str, metavar="MODEL", help="Name of model to train")

    parser.add_argument("--adapter_layer", type=int, default=30, metavar="LENGTH", help="the number of adapter layer")

    parser.add_argument("--adapter_len", type=int, default=10, metavar="LENGTH", help="the adapter length")

    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_path", default="/instruction_dataset/", type=str, help="dataset path")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = InstructionDataset(
        data_path=args.data_path, model_path=args.llama_model_path, max_words=args.max_seq_len, partition="train"
    )
    dataset_val = InstructionDataset(
        data_path=args.data_path, model_path=args.llama_model_path, max_words=args.max_seq_len, partition="val"
    )

    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    for i in range(0, 32):
        args.adapter_layer = i
        # define the model
        model = models_llama_adapter.__dict__[args.model](args)

        model.to(device)

        model_without_ddp = model
        print("Model = %s" % str(model_without_ddp))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if wandb.run is not None:
            wandb.finish()
        wandb.init(project='llama_adapter', name=f'llama-2-7b-toxic_{args.adapter_layer}')

        output_dir = f"{args.output_dir}/toxic_llama-2-7b_{args.adapter_layer}/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Start training for {args.epochs} epochs, layer {args.adapter_layer}")
        start_time = time.time()
        global_step = 0 
        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
                data_loader_val.sampler.set_epoch(epoch)

            train_stats, global_step = train_one_epoch(
                model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args, global_step=global_step, val_dl=data_loader_val
            )

            val_stats, global_step = val_one_epoch(
                model, data_loader_val, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args, global_step=global_step
            )

            if args.output_dir and  epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    save_path = output_dir
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                **{f"val_{k}": v for k, v in val_stats.items()},
            }

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    
    main(args)
