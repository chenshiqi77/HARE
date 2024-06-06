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

from torch.utils.data import Dataset


class GetData(Dataset):
    def __init__(self, dataset, tokenizer, start):
        super(GetData, self).__init__()
        self.dataset = dataset
        self.length = len(dataset)
        self.tokenzier = tokenizer
        self.tokenzier.pad_token = self.tokenzier.unk_token
        self.tokenzier.padding_side = "right"
        self.tokenzier.add_bos_token = True
        self.tokenzier.add_eos_token = True
        self.start = start

    def __len__(self):
        return self.length

    def preprocessing(self, example):
        token = self.tokenzier(
            example["content"],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        input_ids = token["input_ids"][0]
        attention_mask = token["attention_mask"][0]
        labels = input_ids

        return {
            "input_ids": input_ids.to("cuda"),
            "attention_mask": attention_mask.to("cuda"),
            "labels": labels.to("cuda"),
        }

    def __getitem__(self, idx):
        idx += self.start
        return self.preprocessing(self.dataset[idx])
