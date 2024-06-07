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
import os
import numpy as np
import torch
import transformer_engine.pytorch as te
from time import perf_counter
from torch import nn
from transformer_engine.common import recipe
import torch.nn as nn
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

batch_sizes = [1, 1, 8, 8, 1, 8, 8, 1]
hidden_sizes = [1024, 1024, 2048, 2048, 1024, 2048, 2048, 1024]
output_sizes = [1024, 1024, 512, 5632, 1024, 2048, 32, 1024]

fp8_recipe = recipe.DelayedScaling(
    margin=0, interval=1, fp8_format=recipe.Format.HYBRID
)
fp8_recipe.reduce_amax = False
use_te_fp8s = [[False, False], [True, True]]


class Model(torch.nn.Module):
    def __init__(self, hidden_size, output_size, use_te):
        super().__init__()
        if not use_te:
            self.linear = nn.Linear(hidden_size, output_size, bias=False)
        else:
            self.linear = te.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        for i in range(1):
            x = self.linear(x)
        return x


# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器，传入模型的参数和学习率


for i in range(len(batch_sizes)):
    for use_te_fp8 in use_te_fp8s:
        use_te = use_te_fp8[0]
        use_fp8 = use_te_fp8[1]
        batch_size = batch_sizes[i]
        hidden_size = hidden_sizes[i]
        output_size = output_sizes[i]
        model = Model(hidden_size, output_size, use_te)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.to("cuda")
        input = torch.randn(batch_size, hidden_size, hidden_size).to("cuda")
        targets = torch.randn(batch_size, hidden_size, output_size).to("cuda")
        for j in range(5):
            # warmup
            with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
                out = model(input)
        torch.cuda.synchronize()
        N = 22
        start = perf_counter()
        for j in range(N):
            with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
                re = torch.sum(out)
                loss = criterion(re, torch.Tensor([1]).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                out = model(input)
        torch.cuda.synchronize()
        end = perf_counter()
        time = (end - start) / N
        print(
            f"use_te: {str(use_te):5}, use_fp8: {str(use_fp8):5}, bs: {batch_size:5}, hid_size: {hidden_size:5}, out_size: {output_size:5}, cost_time: {1000*time:5f}ms"
        )
    print("=============================================")
