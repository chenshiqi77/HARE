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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """

    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(
        metadata={
            "help": "训练集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"
        }
    )
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    template_name: str = field(default="", metadata={"help": "sft时的数据格式"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    max_prompt_length: int = field(
        default=512, metadata={"help": "dpo时，prompt的最大长度"}
    )
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss"})
    tokenize_num_workers: int = field(
        default=10, metadata={"help": "预训练时tokenize的线程数量"}
    )
    task_type: str = field(
        default="sft", metadata={"help": "预训练任务：[pretrain, sft]"}
    )
    train_mode: str = field(
        default="qlora", metadata={"help": "训练方式：[full, qlora]"}
    )
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "lora dropout"}
    )
    task_model: str = field(default="", metadata={"help": "训练模型：[st, other]"})
    teacher_model_path: str = field(
        default="models/mistral-7b-base", metadata={"help": "训练模型：[st, other]"}
    )
    student_model_path: str = field(
        default="./mistral", metadata={"help": "训练模型：[st, other]"}
    )
    cache_path: str = field(default=None, metadata={"help": "数据加载缓存"})
    eval_dataset_path: str = field(
        default="/home/mutil/datasets/final_arrowdata/UltraChat_new/UltraChat_new.jsonl",
        metadata={"help": "eval数据加载路径"},
    )
    eval_step: Optional[int] = field(default=200, metadata={"help": "eval_step"})
    gradient_clipping: float = field(
        default=1.0, metadata={"help": "gradient_clipping"}
    )
    log_path: str = field(
        default="/root/MutilNode/distillation/log",
        metadata={"help": "tensorboard日志路径"},
    )
    model_output_dir: str = field(
        default="/home/mutil/train_model", metadata={"help": "模型输出路径"}
    )
    num_grad_accum_steps: Optional[int] = field(
        default=4, metadata={"help": "梯度累计步数"}
    )
    offload: bool = field(default=False, metadata={"help": "是否CPUAdam"})
    resume: bool = field(default=True, metadata={"help": "是否续训"})
    resume_model_path: str = field(
        default="/home/mutil/train_model/checkpoint-193800",
        metadata={"help": "模型续训路径"},
    )
    train_dype: str = field(default="fp8", metadata={"help": "训练类型"})
    saving_steps: Optional[int] = field(default=600, metadata={"help": "saving_steps"})
    use_cache: bool = field(default=False, metadata={"help": "是否使用缓存"})
    version: Optional[int] = field(default=6, metadata={"help": "version"})
    warmup_proportion: float = field(
        default=0.0, metadata={"help": "warmup_proportion"}
    )
    data_config: str = field(
        default="./train_args/data_info.ini", metadata={"help": "数据配置加载"}
    )
    data_path: str = field(default="./data", metadata={"help": "数据集路径"})
