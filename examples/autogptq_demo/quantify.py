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

"""
This code is used for quantize fp16 model into 4-bit or 8-bit by auto_gpt.
Reference: https://github.com/AutoGPTQ/AutoGPTQ

Note:
1. Install torch & auto_gptq with cuda before runing this code.
2. Quantization require examples for model compression.

Run this scripts for quantization:
python quantize_model_gptq.py --original_model_path=model_to_be_quantized --quantization_model_path=quantization_model_save_path --quantization=8

"""
import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ================
#    constant
# ================
EXAMPLES = [
    "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
]


# ================
#    functions
# ================
def get_args():
    parser = argparse.ArgumentParser(description="Model quantization with auto_gptq")
    parser.add_argument(
        "--original_model_path", type=str, help="model to be quantized."
    )
    parser.add_argument(
        "--quantization_model_path",
        type=str,
        default=None,
        help="save path for quantized model.",
    )
    parser.add_argument(
        "--quantization", type=int, default=4, help="4-bit or 8-bit quantization."
    )
    parser.add_argument(
        "--examples_path",
        type=str,
        default=None,
        help="examples for model quantization.",
    )
    return parser


def load_datas(path):
    with open(path, "r") as file:
        datas = [json.loads(data)["text"] for data in file]
    return datas


# ================
#      main
# ================
if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    print(args)

    # loading examples for quantization
    if (args.examples_path is not None) and (os.path.exists(args.examples_path)):
        try:
            EXAMPLES = load_datas(args.examples_path)
        except Exception as e:
            pass
    print("Examples for quantization: {}".format(len(EXAMPLES)))

    # loading tokenizer & tokenize examples
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_path)
    examples = []
    for example in tqdm(EXAMPLES):
        examples.append(tokenizer(example))

    # quantization config
    quantize_config = BaseQuantizeConfig(
        bits=int(args.quantization), group_size=128, desc_act=False
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(
        args.original_model_path, quantize_config
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples)

    if args.quantization_model_path is None:
        save_path = args.original_model_path + "-{}bit".format(args.quantization)
    else:
        save_path = args.quantization_model_path
    # save quantized model using safetensors
    model.save_quantized(save_path, use_safetensors=True)
    tokenizer.save_pretrained(save_path)
    print(
        "Model quantization finished! {}bit model saved in `{}`!".format(
            args.quantization, save_path
        )
    )
