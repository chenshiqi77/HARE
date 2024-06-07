# coding=utf-8
# Copyright 2024 Lite-AI Inc. team. All rights reserved.
#
# This code is based on MistralForCausalLM from transformers.
# It has been modified from its original forms to accommodate
# minor architectural differences compared to.
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

import argparse
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoConfig,
)
from utils.argument import CustomizedArguments
from utils.trainer import MyTrainer
from datasets import load_dataset, load_from_disk
from deepspeed.ops.adam import FusedAdam
import json


def set_args():
    args_parse = argparse.ArgumentParser()
    args_parse.add_argument(
        "--train_args_file",
        type=str,
        default="train_args/train_config.json",
        help="",
    )
    args_parse.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )
    parse_args = args_parse.parse_args()
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    cust_args, train_args = parser.parse_json_file(json_file=parse_args.train_args_file)
    # 把transformer的训练参数赋值给cust_args_args
    for attr, value in train_args.__dict__.items():
        setattr(cust_args, attr, value)
    return cust_args


def main():
    args = set_args()

    model_path = args.model_name_or_path
    data_path = args.data_path
    # 模型加载
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 数据加载
    data = load_dataset(data_path, split="train", cache_dir="cachedir")

    deepspeed_config_path = args.deepspeed
    deepspeed_config = json.load(open(deepspeed_config_path, "r"))

    train_object = MyTrainer(
        model=model,
        train_data=data,
        tokenizer=tokenizer,
        num_train_epochs=args.num_train_epochs,
        seed=66,
        per_device_train_batch_size=args.per_device_train_batch_size,
        optimizer=FusedAdam,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_proportion=args.warmup_proportion,
        num_grad_accum_steps=args.num_grad_accum_steps,
        deepspeed_config=deepspeed_config,
        log_path=args.log_path,
        max_saving_checkpoints=args.saving_steps,
        model_output_path=args.output_dir,
        args=args,
        resume=args.resume,
        resume_model_path=args.resume_model_path,
        model_config=config,
    )

    train_object.train()


if __name__ == "__main__":
    main()
