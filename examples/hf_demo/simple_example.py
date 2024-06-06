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
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/lite-ai/HARE-1.1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

prompt = """What is the tallest mountain in the world? How high is it?"""
tokens = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').to(device)
output = model.generate(**tokens)

output_tokens = output[0].cpu().numpy()[tokens.input_ids.size()[1]:]
output_string = tokenizer.decode(output_tokens)
print(output_string)