# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import argparse
import gc
import os
import pickle as pkl
import pprint
import random
import shutil
from contextlib import redirect_stdout
from datetime import datetime
from time import time

import roboverse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# import tqdm
from torch import autocast
# 增加了FSDP的逻辑
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from rv_train import models
from rv_train.configs import get_cfg_defaults
from rv_train.utils import train_utils as utils

DEVICE = ""

START_TIME = time()


def save_checkpoint(name, epoch, model, optimizer, lr_sched, cfg, log_dir):
    """
    Saves all information required for resuming training in the experiment
    folder.
    """
    # take care of DDP
    if isinstance(model, DDP):
        model_module = model.module
    else:
        model_module = model

    # take care of model saving for models that have save_pretrained method
    if hasattr(model_module, "save_pretrained"):
        model_state = None
        print("WARNING: model has save_pretrained method, not saving model state")
        model_module.save_pretrained(f"{log_dir}/model_{name}")
    else:
        model_state = model_module.state_dict()

    # Prepare checkpoint data
    # 如果是 HF 模型：这里是 None，因为你已经用 save_pretrained 存到目录里了；如果是普通模型：这里是 state_dict().
    checkpoint_data = {
        "cfg": vars(cfg),  # vars(cfg) 会把它转成普通的 dict（键值为所有字段）。
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "lr_sched_state": lr_sched.state_dict() if lr_sched is not None else None,
    }

    pth_path = f"{log_dir}/model_{name}.pth"
    torch.save(checkpoint_data, pth_path)

    # save the dataset stats
    """
    这段是跟你当前工程（比如 VLA-0）强相关的逻辑：
        cfg.EXP.MODEL 指明本次实验的模型类型，例如：
            "qwen"：纯 Qwen 模型？
            "dp"：diffusion policy？
            "qwen_dp"：两者混合？
        对这几种模型，代码约定：模型内部有一个属性：
        model_module.original_dataset_stats
        典型用途：训练前根据数据集统计得到的一些归一化信息：
            均值 / 方差；
            action / state 范围；
            其他统计量（例如用于 rescale 输出）。
        这里用 pickle 单独存为：
        {log_dir}/dataset_stats.pkl
        便于：
            以后单独加载推理用（比如只有一个 ckpt 没有全项目代码时，也能拿到 stats）；
            或者 eval 脚本直接用 log_dir/dataset_stats.pkl 做归一化。
    也就是说，此处除了 .pth，又额外保存了一个数据集统计文件。
    """
    if cfg.EXP.MODEL in ["qwen", "dp", "qwen_dp"]:
        with open(f"{log_dir}/dataset_stats.pkl", "wb") as f:
            pkl.dump(model_module.original_dataset_stats, f)

    print(f"Checkpoint saved to {pth_path}.")


# def load_model(model, model_path, cfg):
#     """
#     Loads a pretrained model from a given path.
#     :param model: model to load
#     :param model_path: path to the pretrained model
#     :param cfg: config object
#     """
#     print(f"Recovering model and checkpoint from {model_path}")
#     checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

#     # take care of DDP
#     if isinstance(model, DDP):
#         model_module = model.module
#     else:
#         model_module = model

#     # take care of model loading for models that have load_pretrained method
#     if hasattr(model_module, "from_pretrained"):
#         print("WARNING: model has from_pretrained method")
#         assert model_path[-4:] == ".pth"
#         print(f"Loading from {model_path[:-4]}")
#         model_module.from_pretrained(model_path[:-4])
#     else:
#         model_module.load_state_dict(checkpoint["model_state"])

#     # load the dataset stats
#     if cfg.EXP.MODEL in ["qwen", "dp", "qwen_dp"]:
#         log_dir = "/".join(model_path.split("/")[:-1])
#         with open(f"{log_dir}/dataset_stats.pkl", "rb") as f:
#             original_dataset_stats = pkl.load(f)
#             model_module.set_dataset_stats(original_dataset_stats)


#     return model, checkpoint
def load_model(model, model_path, cfg):
    """
    Loads a pretrained model from a given path.
    :param model: model to load
    :param model_path: path to the pretrained model
    :param cfg: config object
    """
    print(f"Recovering model and checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # 兼容 DDP / FSDP
    if isinstance(model, (DDP, FSDP)):
        model_module = model.module
    else:
        model_module = model

    # 带 from_pretrained 的特殊模型
    if hasattr(model_module, "from_pretrained"):
        print("WARNING: model has from_pretrained method")
        assert model_path[-4:] == ".pth"
        print(f"Loading from {model_path[:-4]}")
        model_module.from_pretrained(model_path[:-4])
    else:
        state = checkpoint["model_state"]
        if isinstance(model, FSDP):
            # FSDP: 用 FULL_STATE_DICT 加载，内部自动切片
            full_cfg = FullStateDictConfig(rank0_only=False, offload_to_cpu=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
                model.load_state_dict(state)
        else:
            model_module.load_state_dict(state)

    # load the dataset stats（log_dir 推断逻辑不变）
    if cfg.EXP.MODEL in ["qwen", "dp", "qwen_dp"]:
        log_dir = "/".join(model_path.split("/")[:-1])
        with open(f"{log_dir}/dataset_stats.pkl", "rb") as f:
            original_dataset_stats = pkl.load(f)
            model_module.set_dataset_stats(original_dataset_stats)

    return model, checkpoint


def load_model_opt_sched(
    model,
    optimizer,
    lr_sched,
    model_path,
    cfg,
    to_load_model=True,
    only_load_model=False,
):
    """
    在 load_model 基础上，加了一层策略开关
    模型 + 优化器 + 调度器 + epoch 全家桶恢复”的工具函数。
    Loads a pretrained model from a given path.
    :param model: model to load
    :param optimizer: optimizer to load
    :param lr_sched: learning rate scheduler to load
    :param model_path: path to the pretrained model
    :param cfg: config object
    :param to_load_model: whether to load the model from the checkpoint or not
    """
    if to_load_model:
        model, checkpoint = load_model(model, model_path, cfg)
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if not only_load_model:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if lr_sched is not None:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])

    epoch = checkpoint["epoch"]

    # clean GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    return model, epoch, optimizer, lr_sched


def get_pretrained_model(model_path, device, torch_compile=False):
    """
    Loads a pretrained model from a given path.
    :param model_path: path to the pretrained model
    :param device: device to load the model on, supports only single GPU for now
    :return: model, cfg
    👉 “给你一个 ckpt 路径，我帮你把 config、模型构建、权重加载、dataset_stats 注入、device 迁移、可选 compile 一次性都做好，直接拿来推理或当预训练初始化”。
    """
    model_folder = "/".join(model_path.split("/")[:-1])
    cfg_path = model_folder + "/config.yaml"
    cfg = get_cfg(cfg_path, cfg_opts="")

    model = get_model(
        cfg, calculate_dataset_stats=False
    )  # don't calculate dataset stats for pretrained model, its loaded from a checkpoint
    # model.to(device) devicemap auto后不用自己管理
    optimizer, lr_sched = get_optimizer(cfg, model, num_gpus=1)

    model, _, _, _ = load_model_opt_sched(
        model=model,
        optimizer=optimizer,
        lr_sched=lr_sched,
        model_path=model_path,
        cfg=cfg,
        only_load_model=True,
    )

    if torch_compile:
        print(
            "Compiling model with torch.compile, this will put the model in eval mode and may take a while..."
        )
        model.eval()
        model = torch.compile(model)
        if hasattr(model, "model"):
            if hasattr(model.model, "generate"):
                print("Compiling model.model.generate with torch.compile")
                model.model.generate = torch.compile(model.model.generate)

    return model, cfg


def get_cfg(cfg_path, cfg_opts):
    # 👉 从默认 + 配置文件 + 可选命令行覆盖，生成一个“冻结”的配置对象。
    cfg = get_cfg_defaults()
    if cfg_path != "":
        cfg.merge_from_file(cfg_path)

    if cfg_opts != "":
        cfg.merge_from_list(cfg_opts.split(" "))
        cfg.EXP.EXP_ID += f"_{utils.short_name(cfg_opts)}"
    cfg.freeze()

    print(cfg)
    return cfg


def get_inp(cfg, data_batch):
    """
    Constructs the input for the model using the batched data.
    :param cfg: config object
    :param data_batch: contains the batched data provided by the dataloader
    现在的 get_inp 是“空适配层”，啥也不干，只是原样返回 data_batch，但它作为接口存在，是为了以后在这里集中实现「数据 batch → 模型输入」的所有转换逻辑
    """

    inp = data_batch
    return inp


def get_model(cfg, calculate_dataset_stats=True):
    """
    Returns model based on the config
    """
    if cfg.EXP.MODEL == "qwen":
        model = models.QwenActor(**cfg.MODEL.QWEN)
    else:
        assert False, f"Invalid model: {cfg.EXP.MODEL}"

    if calculate_dataset_stats and cfg.EXP.MODEL in ["qwen"]:
        temp_dataset = get_dataloader(split="train", cfg=cfg, get_dataset=True)
        model.set_dataset_stats(temp_dataset.stats)
        del temp_dataset

    return model


def default_batch_proc(data_batch, device):
    for x in data_batch:
        if isinstance(data_batch[x], dict):
            for y in data_batch[x]:
                data_batch[x][y] = data_batch[x][y].to(device).float()
        else:
            if isinstance(data_batch[x], torch.Tensor):
                data_batch[x] = data_batch[x].to(device).float()
            else:
                data_batch[x] = data_batch[x]
    return data_batch


def get_dataloader(split, cfg, get_dataset=False):
    """
    Returns dataloader based on the config and split
    :param get_dataset: whether to return the dataset or the dataloader
    """
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size
    dataset_args = {"split": split}

    if cfg.EXP.DATASET == "roboverse":
        print("WARNING: split is ignored for roboverse dataset.")
        dataset_args = dict(**cfg.DATALOADER.ROBOVERSE)
        dataset = roboverse.get_unified_dataset(**dataset_args)
    else:
        raise NotImplementedError

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = default_batch_proc

    if get_dataset:
        return dataset
    else:
        """
        标准 ImageNet DDP 的做法是：用 DistributedSampler 按 rank 切分数据 → 每个进程看 不同子集；
        而这份代码如果 roboverse.get_unified_dataset 内部没有做 per-rank 切分，
        那就是：每个进程都在整个数据集上跑一遍，只是顺序不一样。
        RoboVerse 的 dataset 内部自己做了 rank/world_size 切分?等我看下get_unified_dataset
        """
        return DataLoader(
            dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            pin_memory=(torch.cuda.is_available()) and (not num_workers),
            persistent_workers=(num_workers > 0),
        )


def check_grad(model, loss):
    bad_grad = False
    if loss.ne(loss).any():
        bad_grad = True
        print("WARNING: nan in the loss")
    else:
        for x in model.parameters():
            if x.grad is not None:
                if x.grad.ne(x.grad).any():
                    print("WARNING: nan in a gradient")
                    bad_grad = True
                    break
                if ((x.grad == float("inf")) | (x.grad == float("-inf"))).any():
                    print("WARNING: inf in a gradient")
                    bad_grad = True
                    break
    return bad_grad


def get_optimizer(cfg, model, num_gpus=1):
    """
    Returns optimizer and learning rate scheduler based on the config
    :param cfg: config object
    :param model: model to optimize
    :param num_gpus: number of GPUs to optimize the model on, required for scaling the learning rate
    """
    if cfg.EXP.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adam_bnb":
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adamw_bnb":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adamw_bnb_fp8":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    else:
        raise NotImplementedError

    if cfg.EXP.LR_SCHED == "none":
        lr_sched = None
    elif cfg.EXP.LR_SCHED == "cosine_anneal":
        lr_sched = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.TRAIN.num_epochs, eta_min=cfg.LR_SCHED.lr_clip
        )
    else:
        raise NotImplementedError

    return optimizer, lr_sched


def train(
    cfg,
    loader,
    model,
    optimizer,
    device=0,
    check_grad_fn=False,  # Rename this parameter
    fn_check_time_limit_and_relaunch=None,
    rank=0,
    epoch=0,
    tb=None,
):
    """
    Training for one epoch
    """

    model.train()
    # 创建性能跟踪器
    perf = utils.PerfTrackTrain(cfg)

    LOG_EVERY = 20  # 每 20 个 batch 打一行印
    global_step_base = epoch * len(loader)  # 用于计算全局 step
    TB_EVERY = 1  # 每 1 个 batch 往 TB 写一次

    time_for = 0  # 前向传播总时间
    time_bac = 0  # 反向传播总时间
    time_dl = 0  # 数据加载总时间
    time4 = time()  # 记录循环开始时间（用于计算数据加载时间）

    epoch_start_time = time()  # 用来算每迭代耗时
    num_batches = len(loader)  # 总共多少个 iter

    # 进度条宽度自适应终端宽度
    # for i, data_batch in tqdm.tqdm(enumerate(loader), dynamic_ncols=True):
    for i, data_batch in enumerate(loader):
        # 对batch数据进行预处理（如数据增强）batch processing
        data_batch = loader.dataset.batch_proc(data_batch, device)
        # 从batch中提取模型需要的输入
        inp = get_inp(cfg, data_batch)

        time1 = time()
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.EXP.AMP):
            out = model(**inp, get_loss=True)
        loss = out["loss"]
        perf.update_all(data_batch=data_batch, out=out, loss=loss)

        time2 = time()  # 记录反向传播开始时间
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        # 梯度裁剪：防止梯度爆炸，将梯度范数限制在cfg.TRAIN.clip_grad_norm
        if cfg.TRAIN.clip_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.clip_grad_norm)

        if check_grad_fn and check_grad(model, loss):  # Use the renamed parameter
            print("WARNING: avoiding step as bad gradient")
        else:
            optimizer.step()

        time3 = time()
        time_dl += time1 - time4
        time_for += time2 - time1
        time_bac += time3 - time2
        time4 = time()
        # =============================================================
        # ====== ⭐ 新增：每 N 个 batch 打印一行 ======
        curr_loss = loss.item()
        avg_loss = perf.agg_loss()
        if rank == 0 and ((i + 1) % LOG_EVERY == 0 or i == 0):
            now = time()
            elapsed = now - epoch_start_time
            it_per_sec = (i + 1) / max(elapsed, 1e-6)

            print(
                f"[epoch {epoch}/{cfg.TRAIN.num_epochs - 1}] "
                f"iter {i+1}/{num_batches}  "
                f"loss {curr_loss:.4f}  avg_loss {avg_loss:.4f}  "
                f"it/s {it_per_sec:.2f}  "
                f"t_fwd {time_for/(i+1):.3f}s  "
                f"t_bwd {time_bac/(i+1):.3f}s  "
                f"t_data {time_dl/(i+1):.3f}s"
            )
        if rank == 0 and tb is not None and (i + 1) % TB_EVERY == 0:
            # ======= ⭐ 可选：写入 TensorBoard，一样只在 rank 0 做 =======
            global_step = global_step_base + i + 1
            tb.update(
                "train_iter",
                global_step,
                {"loss": curr_loss, "avg_loss": avg_loss},
            )
        """
        GPU 服务器、云平台、学校集群（Slurm/HTCondor）都有“时间限制”。
        比如一个训练任务最多只能跑 8 小时，超过会被系统强制杀掉。
        训练一段时间（例如 7 小时 50 分）
        快到时间上限 → 自动保存模型 checkpoint
        优雅退出（不让系统强制杀）
        自动重新启动相同的训练脚本
        从刚刚保存的 checkpoint 继续训练
        """
        if fn_check_time_limit_and_relaunch is not None:
            # checking every 300 batches ~ 5 minutes
            if (i + 1) % 300 == 0:
                fn_check_time_limit_and_relaunch(perf.agg_loss())

        # uncomment for intermediate printing
        # if i % 10 == 0:
        #     print(f"Iteration {i} time taken: {time_for:.2f}s, {time_bac:.2f}s, {time_dl:.2f}s")

    print(
        f"Avg_loss: {perf.agg_loss():.4f}, "
        f"Forward: {time_for:.2f}s, Backward: {time_bac:.2f}s, "
        f"Data Load: {time_dl:.2f}s, "
        f"Memory Usage: {utils.get_gpu_memory_map()}"
    )

    return perf.agg(), perf.agg_loss()


def print_model_stats(model):
    """Print model statistics including parameter counts."""
    # Get model module if using DDP
    model_module = model.module if isinstance(model, (DDP, FSDP)) else model

    # Count total parameters
    total_params = sum(p.numel() for p in model_module.parameters())

    # Count trainable parameters
    trainable_params = sum(
        p.numel() for p in model_module.parameters() if p.requires_grad
    )

    # Count non-trainable parameters
    non_trainable_params = total_params - trainable_params

    print("=" * 50)
    print("Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 50)


def get_log_dir(cfg, logdir_with_time=False):
    if logdir_with_time:
        log_dir = (
            f"./runs/{cfg.EXP.EXP_ID}/{str(datetime.now())[:-7].replace(' ', '-')}"
        )
    else:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    return log_dir


def entry_train(
    rank,  # 当前进程的 rank（0,1,2,...），用来做 DDP、多卡分工。
    cfg,
    logdir_with_time=False,
    resume=False,
    model_path="",
    devices=[0],
    port=12345,
):
    """
    Training and evaluating a network based on the specified config.
    """

    # 取出当前 rank 对应的 GPU ID
    device = devices[rank]
    device = f"cuda:{device}"
    # 如果 devices 列表长度 > 1，就说明要用 DDP（多卡）。
    ddp = len(devices) > 1
    # 初始化进程组（dist.init_process_group）。
    # 把所有 rank 进程连成一条通信线，用于梯度同步等。
    # 所有参与分布式训练的进程都要各自调用 init_process_group
    utils.setup(rank, world_size=len(devices), port=port)
    torch.cuda.set_device(device)
    if ddp:
        print(f"Running on rank {rank}")

    # 理论上每个 rank 会用 SEED + rank 做随机种子，保证分布式时乱数不同、又可复现。
    # 现在是注释状态，说明作者暂时不用这个（可能统一在别处设种子）。
    # random.seed(cfg.EXP.SEED + rank)
    # np.random.seed(cfg.EXP.SEED + rank)
    # torch.manual_seed(cfg.EXP.SEED + rank)

    loader_train = get_dataloader(split="train", cfg=cfg)
    model = get_model(cfg)
    # model.to(device) # devicemap auto后不用自己管理

    # FSDP 增加
    model.device = device

    # 默认 to_load_model = True，表示后面会通过 load_model_opt_sched 加载参数
    to_load_model = True
    if (
        hasattr(model, "load_param_before_ddp")
        and model.load_param_before_ddp
        and resume
    ):
        to_load_model = False
        # 有的模型在包进 DDP 前加载更安全，比如自定义 module、冻结部分参数。
        model, _ = load_model(model, model_path, cfg)
        # device map auto后不用自己管理
        # model.to(device)

    if ddp:
        # Set find_unused_parameters=False when using gradient checkpointing
        # to avoid synchronization issues and deadlocks
        # 梯度检查点：一种内存优化技术，通过牺牲计算时间（重新计算前向传播）来减少内存占用
        # using_grad_checkpoint = False
        using_grad_checkpoint = True
        if cfg.EXP.MODEL in ["qwen", "qwen_dp"]:
            model_config = (
                cfg.MODEL.QWEN if cfg.EXP.MODEL == "qwen" else cfg.MODEL.QWEN_DP
            )
            # using_grad_checkpoint = getattr(model_config, "grad_checkpoint", False)
            using_grad_checkpoint = getattr(model_config, "grad_checkpoint", True)

        # 这是分布式数据并行（DDP）中的一个重要参数，用于控制是否查找未使用的参数。
        # 默认情况下（无梯度检查点）：开启检测，确保训练正确性
        # 使用梯度检查点时：关闭检测，避免与检查点机制冲突
        find_unused_params = not using_grad_checkpoint
        if rank == 0:
            print(
                f"DDP configuration: grad_checkpoint={using_grad_checkpoint}, find_unused_parameters={find_unused_params}"
            )
        # model = DDP(
        #     model, device_ids=[device], find_unused_parameters=find_unused_params
        # )
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # FULL_SHARD: 参数 + 梯度 + optimizer 状态全部切片
        # 启用lora时候崩溃，FSDP要求一个张量里所有元素必须是相同数据类型 暂时先不处理
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            mixed_precision=mp_policy,
            device_id=device,
            use_orig_params=True,
        )
    if rank == 0:
        print(model)

    optimizer, lr_sched = get_optimizer(cfg, model, num_gpus=len(devices))
    if resume:
        model, old_epoch, optimizer, lr_sched = load_model_opt_sched(
            model=model,
            optimizer=optimizer,
            lr_sched=lr_sched,
            model_path=model_path,
            cfg=cfg,
            to_load_model=to_load_model,
        )
    else:
        assert model_path == "", model_path
        old_epoch = -1

    if rank == 0:
        print_model_stats(model)

    # 所有 rank 在这里 集合 一次。
    # 确保模型加载、optimizer 初始化、log_dir 准备等都完成了，再一起进入训练循环。
    dist.barrier()

    # 日志目录 & TensorBoard 只由 rank 0 管
    if rank == 0:
        log_dir = get_log_dir(cfg, logdir_with_time)
        print(f"Log directory: {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(f"{log_dir}/config.yaml", "w") as f:
            with redirect_stdout(f):
                print(cfg.dump())

        # tb is initialized only for rank 0
        tb = utils.TensorboardManager(log_dir)
    else:
        # log_dir and tb should not be used for any rank other than rank 0
        log_dir = ""
        tb = None
    # 主训练循环
    for epoch in range(old_epoch + 1, cfg.TRAIN.num_epochs):
        # fn_check_time_limit_and_relaunch 目前是 None，预留“超时重启”钩子（比如集群 job 限时）。
        fn_check_time_limit_and_relaunch = None

        # print epoch number
        if rank == 0:
            print(f"Training for epoch {epoch} / {cfg.TRAIN.num_epochs}")

        # train
        train_perf, train_loss = train(
            cfg=cfg,
            loader=loader_train,
            model=model,
            optimizer=optimizer,
            device=device,
            fn_check_time_limit_and_relaunch=fn_check_time_limit_and_relaunch,
            rank=rank,
            epoch=epoch,
            tb=tb if rank == 0 else None,
        )

        # update tensorboard
        if rank == 0:
            _lr = (
                lr_sched.optimizer.param_groups[0]["lr"]
                if lr_sched
                else optimizer.param_groups[0]["lr"]
            )
            pprint.pprint(f"Performance: {train_perf}", width=80)
            tb.update("train", epoch, train_perf)
            tb.update(
                "train",
                epoch,
                {"loss": train_loss, "lr": _lr},
            )

        # save checkpoint
        if rank == 0:
            if not (cfg.EXP_EXTRA.save_ckp == 0) and (
                epoch % cfg.EXP_EXTRA.save_ckp == 0
            ):
                save_checkpoint(
                    f"{epoch}",
                    epoch,
                    model,
                    optimizer,
                    lr_sched,
                    cfg,
                    log_dir,
                )

            if cfg.EXP_EXTRA.save_last_ckpt:
                # change name of last checkpoint to second_last so that it is not overwritten by the new last checkpoint.
                # this second last checkpoint will be used to resume training if the training is relaunched because of loss increase
                if os.path.exists(log_dir + "/model_last.pth"):
                    # remove second last checkpoint if it exists
                    if os.path.exists(log_dir + "/model_second_last.pth"):
                        os.remove(log_dir + "/model_second_last.pth")
                    os.rename(
                        log_dir + "/model_last.pth", log_dir + "/model_second_last.pth"
                    )
                if os.path.exists(log_dir + "/model_last"):
                    # remove second last checkpoint if it exists
                    if os.path.exists(log_dir + "/model_second_last"):
                        shutil.rmtree(log_dir + "/model_second_last")
                    os.rename(log_dir + "/model_last", log_dir + "/model_second_last")
                save_checkpoint(
                    "last",
                    epoch,
                    model,
                    optimizer,
                    lr_sched,
                    cfg,
                    log_dir,
                )

        # update learning rate
        if cfg.EXP.LR_SCHED in ["none"]:
            print(f"Current lr: {optimizer.param_groups[0]['lr']}")
        elif cfg.EXP.LR_SCHED in ["cosine_anneal"]:
            lr_sched.step()
            print(f"Current lr: {lr_sched.optimizer.param_groups[0]['lr']}")
        else:
            raise NotImplementedError

    if rank == 0:
        print("Saving the final model")
        save_checkpoint(
            "final",
            cfg.TRAIN.num_epochs - 1,
            model,
            optimizer,
            lr_sched,
            cfg,
            log_dir,
        )

    if rank == 0:
        # close tensorboard
        tb.close()


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cpu":
        print("WARNING: Using CPU")

    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument("--entry", type=str, default="train")
    parser.add_argument("--exp-config", type=str, default="")
    parser.add_argument("--exp-cfg-opts", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--logdir-with-time", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")

    cmd_args = parser.parse_args()

    if cmd_args.entry == "train":
        assert (
            not cmd_args.logdir_with_time
        ), "Temporarily disable logdir_with_time as it is not handled properly when autoresuming and auto-relaunching with loss increase. It is fine for one time launch or manual relaunching."
        _cfg = get_cfg(cmd_args.exp_config, cmd_args.exp_cfg_opts)
        if cmd_args.resume:
            if cmd_args.model_path == "":
                print(
                    "WARNING: No model path provided, resuming from latest checkpoint"
                )
                log_dir = get_log_dir(_cfg, cmd_args.logdir_with_time)
                cmd_args.model_path = os.path.join(log_dir, "model_last.pth")
            print(f"Resuming from {cmd_args.model_path}")
        else:
            assert cmd_args.model_path == ""

        devices = cmd_args.devices.split(",")
        devices = [int(x) for x in devices]
        # 随机在 [27000, 29999] 范围里选一个端口号，用于 DDP 进程间通信（init_process_group）。
        port = (random.randint(0, 3000) % 3000) + 27000
        mp.spawn(
            entry_train,
            args=(
                _cfg,
                cmd_args.logdir_with_time,
                cmd_args.resume,
                cmd_args.model_path,
                devices,
                port,
            ),
            nprocs=len(devices),
            join=True,
        )

    else:
        assert False
