# coding=utf-8
# Copyright 2024 Lite-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on MistralForCausalLM from transformers.
# It has been modified from its original forms to accommodate
# minor architectural differences compared to
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from transformers import get_scheduler
import math
import deepspeed
import torch
import logging
from tqdm.auto import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from typing import Union
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from .dataloader import GetData
import transformer_engine.pytorch as te
from transformer_engine.common import recipe


class AverageLoss:
    def __init__(self, buffer_size=20):
        self.buffer_size = buffer_size
        self.buffer = []

    def empty(self):
        self.buffer = []

    def update(self, loss):
        self.buffer.append(loss)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    @property
    def average(self):
        return sum(self.buffer) / len(self.buffer)


class MyTrainer:
    def __init__(
        self,
        train_data,
        num_train_epochs,
        seed,
        tokenizer,
        per_device_train_batch_size,
        optimizer,
        learning_rate,
        lr_scheduler_type,
        warmup_proportion,
        num_grad_accum_steps,
        deepspeed_config,
        max_saving_checkpoints,
        args,
        model_config,
        model: Union[PreTrainedModel, nn.Module] = None,
        resume: bool = False,
        resume_model_path: str = None,
        model_output_path: str = None,
        use_tensorboard: bool = True,
        log_path: str = None,
        use_deepspeed: bool = True,
    ):
        self.model = model
        self.num_train_epochs = num_train_epochs
        self.use_deepspeed = use_deepspeed
        self.train_data = train_data
        self.seed = seed
        self.resume = resume
        self.per_device_train_batch_size = per_device_train_batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_proportion = warmup_proportion
        self.num_grad_accum_steps = num_grad_accum_steps
        self.deepspeed_config = deepspeed_config
        self.use_tensorboard = use_tensorboard
        self.log_path = log_path
        self.max_saving_checkpoints = max_saving_checkpoints
        self.model_output_path = model_output_path
        self.resume_model_path = resume_model_path
        self.tokenizer = tokenizer
        self.args = args
        self.model_config = model_config

        if self.use_tensorboard:
            assert (
                self.log_path is not None
            ), "tensordboard log_path is needed, you should use log_path = your log path."
            self.summary = SummaryWriter(os.path.join(self.log_path, "runs"))

        self.training_data = GetData(self.train_data, self.tokenizer, 0)
        self.deepspeed_config.update(
            **{
                "train_micro_batch_size_per_gpu": self.per_device_train_batch_size,
                "gradient_accumulation_steps": self.num_grad_accum_steps,
                "gradient_clipping": 1,
            }
        )

    def get_log(self):
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
        return log

    def init_deepspeed(self, train_data):
        sampler = DistributedSampler(
            train_data, seed=self.seed, num_replicas=None, rank=None
        )
        distributed_dataloader = DataLoader(
            train_data,
            batch_size=self.per_device_train_batch_size,
            sampler=sampler,
        )
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(
            self.model, 1e-1
        )

        opti = self.optimizer(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.num_update_steps_per_epoch = math.ceil(
            len(distributed_dataloader) / self.num_grad_accum_steps
        )
        num_warmup_steps = int(
            self.warmup_proportion
            * self.num_train_epochs
            * self.num_update_steps_per_epoch
        )

        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=opti,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_epochs * self.num_update_steps_per_epoch,
        )

        model_parameters = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=model_parameters,
            config=self.deepspeed_config,
            optimizer=opti,
            lr_scheduler=lr_scheduler,
        )
        return distributed_dataloader, model_engine, sampler

    def train(self):
        fp8_recipe = recipe.DelayedScaling(
            margin=0, interval=1, fp8_format=recipe.Format.HYBRID
        )
        fp8_recipe.reduce_amax = False
        log = self.get_log()
        torch.cuda.set_device(self.args.local_rank)
        deepspeed.init_distributed()

        self.args.global_rank = torch.distributed.get_rank()
        main_process = self.args.global_rank == 0

        distributed_dataloader, model_engine, sampler = self.init_deepspeed(
            self.training_data
        )

        if self.resume:
            assert (
                self.resume_model_path is not None
            ), "You must provide a checkpoint path which is needed to be resumed."
            model_engine.load_checkpoint(
                self.resume_model_path,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            previous_step = int(
                os.path.basename(self.args.resume_model_path).split("-")[1]
            )
            all_step = previous_step
            start_step = (all_step - 1) * self.num_grad_accum_steps
            self.training_data = GetData(self.train_data, self.tokenizer, previous_step)
        else:
            all_step = 0
            start_step = 0

        total_batch_size = (
            self.per_device_train_batch_size
            * self.num_grad_accum_steps
            * dist.get_world_size()
        )

        num_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch

        if torch.distributed.get_rank() == 0:
            log.info("***** Running training *****")
            log.info(f"  Num examples = {len(distributed_dataloader)}")
            log.info(f"  Total optimization steps = {num_train_steps}")
            log.info(f"  Gradient accumulation steps = {self.num_grad_accum_steps}")
            log.info(f"  Num epochs = {self.num_train_epochs}")
            log.info(
                f"  Instantaneous batch size per device = {self.per_device_train_batch_size}"
            )
            log.info(
                f"  Total train batch size (w. accumulation, parallel & distributed) = {total_batch_size}"
            )

        bar = tqdm(range(num_train_steps), total=num_train_steps)
        train_losses = AverageLoss(self.num_grad_accum_steps)
        saving_steps = num_train_steps / self.max_saving_checkpoints

        for epoch in range(self.num_train_epochs):
            sampler.set_epoch(epoch)
            for step, batch in enumerate(distributed_dataloader, start=start_step):
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    model_engine.train()
                    s_output = model_engine(**batch)
                    loss = s_output[0]
                    loss_ = loss.detach().clone()
                    dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
                    train_losses.update(loss_.item() / dist.get_world_size())
                    loss = loss / self.num_grad_accum_steps
                    model_engine.backward(loss)
                    torch.cuda.synchronize()
                    model_engine.step()
                if (
                    step % self.num_grad_accum_steps == 0
                    or step == len(distributed_dataloader) - 1
                ):
                    all_step += 1
                    bar.update(1)
                    bar.set_description(f"Epoch [{epoch}/{self.num_train_epochs}]")
                    lr = model_engine.client_lr_scheduler
                    lr = lr.get_last_lr()
                    if main_process:
                        train_all_loss = {"train_loss/train": train_losses.average}
                        self.summary.add_scalars(
                            "train_losses/train", train_all_loss, all_step
                        )
                        self.summary.add_scalar("lr/train", lr[0], all_step)
                    if saving_steps != 0 and all_step % saving_steps == 0:
                        log.info("*************** saving checkpoint... ***************")
                        self.save_model(model_engine, all_step)
                    if main_process:
                        print(
                            f"epoch:{epoch+1}, step:{step}, all_step:{all_step}, total_loss: {train_losses.average}, lr: {lr[0]}"
                        )
            model_engine.tput_timer.update_epoch_count()
        self.save_model(model_engine, "last")

    def save_model(self, model_engine, all_step):
        save_dir = os.path.join(self.model_output_path, f"checkpoint-{all_step}")
        os.makedirs(save_dir, exist_ok=True)
        model_engine.save_checkpoint(save_dir)
        self.model_config.save_pretrained(save_dir)
        state_dict = model_engine.module.state_dict()
        self.tokenizer.save_pretrained(save_dir)
        save_dict = {}
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            save_dict[k] = v
        torch.save(save_dict, os.path.join(save_dir, "pytorch_model.bin"))

    def get_optimizer_grouped_parameters(self, model, weight_decay):
        no_decay_name_list = [
            "bias",
            "LayerNorm",
            "ln",
            "layer_norm",
            "layernorm",
            "norm",
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (
                        not any(nd in n for nd in no_decay_name_list)
                        and p.requires_grad
                    )
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
