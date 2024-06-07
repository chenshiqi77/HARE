# coding=utf-8

import torch
import transformer_engine.pytorch as te

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_engine.common import recipe

# ======================
#       main
# ======================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp8_recipe = recipe.DelayedScaling(
        margin=0, interval=1, fp8_format=recipe.Format.HYBRID
    )
    fp8_recipe.reduce_amax = False

    model_path = "LiteAI-Team/Hare-1.1B-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    prompt = "Write a poem based on the landscape of Guizhou:"
    tokens = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(device)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        output = model.generate(**tokens)

    output_tokens = output[0].cpu().numpy()[tokens.input_ids.size()[1] :]
    output_string = tokenizer.decode(output_tokens)
    print(output_string)
