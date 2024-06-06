# -*- coding: utf-8 -*-
# fastapi + gradio 搭建web ui

import torch
import gradio as gr

from fastapi import FastAPI
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings


app = FastAPI()

# ======================
#       variables
# ======================
class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, token_id_list):
        self.token_id_list = token_id_list
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

# prompt
SYSTEM_PROMPT = "You are a helpful assistant."
stopping_criteria = StoppingCriteriaList()
stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[32001]))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "LiteAI-Team/Hare-1.1B-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

# ======================
#       functions
# ======================
def chat(
    messages,
    model,
    tokenizer,
    top_p=0.8,
    temperature=0.8,
    max_length=1024,
    max_new_tokens=256
):    
    n_token = max_length
    system = "<round_start>system\n{}<round_end>\n".format(SYSTEM_PROMPT)
    system_token = tokenizer.encode(system, add_special_tokens=False)
    n_token -= len(system_token)

    query = messages[-1][0]
    query = "<round_start>user\n{}<round_end>\n<round_start>assistant\n".format(query)
    query_token = tokenizer.encode(query, add_special_tokens=False)
    n_token -= len(query_token)
    
    messages = messages[:-1]
    conversations = []
    for ids in range(len(messages)-1, 0, -1):
        conv = messages[ids]
        user = conv[0]
        assistant = conv[1]
        
        round = "<round_start>user\n{}<round_end>\n<round_start>assistant\n{}<round_end>\n".format(user, assistant)
        round_token = tokenizer.encode(round, add_special_tokens=False)

        if n_token - len(round_token) > 0:
            conversations = [round] + conversations
        else:
            break

    prompt = system + "".join(conversations) + query
    prompt_token = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_token.to(model.device)

    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(
        prompt_token,
        do_sample=True,
        top_k=0,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    answer = ""
    for new_text in streamer:
        answer += new_text
        yield answer[4 + len(prompt):]


# ======================
#     demo functions
# ====================== 
def change_prompt(inputs):
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = inputs
    print(" ********* Re-write system prompt ! ********* ")
    print(SYSTEM_PROMPT)
    return inputs


def predict(_query, _chatbot, _task_history, top_p, temperature, max_tokens):
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        
        for new_text in chat(
            messages=_chatbot,
            model=model,
            tokenizer=tokenizer,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_tokens
        ):
            if new_text is not None:
                response = new_text
                _chatbot[-1] = (_query, response)

            yield _chatbot
            full_response = response            

        # print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Response: {full_response}")


def regenerate(_chatbot, _task_history):
    if not _task_history:
        yield _chatbot
        return
    item = _task_history.pop(-1)
    _chatbot.pop(-1)
    yield from predict(item[0], _chatbot, _task_history)


def reset_user_input():
    return gr.update(value="")


def reset_state(_chatbot, _task_history):
    _task_history.clear()
    _chatbot.clear()
    return _chatbot

# ======================
#  main gradio blocks
# ====================== 
@app.get('/')
async def root():
    return 'Gradio app is running at /gradio', 200

with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=3>Try Models</center>""")
    # 分左右两栏展示
    with gr.Row():
        with gr.Column(scale=0.2):    
            with gr.Accordion("Generate Configurations", open=True) as generate_configs:
                max_tokens = gr.Slider(0, 1024, value=64, step=64, label="max_tokens")
                temperature = gr.Slider(0, 1, value=0.8, step=0.01, label="temperature")
                top_p = gr.Slider(0, 1, value=0.8, step=0.1, label="top_p")

        with gr.Column(scale=1):
            with gr.Accordion("Prompt") as prompt_settings:
                prompts = gr.Textbox(value=SYSTEM_PROMPT, label="[Prompt指令]", placeholder=SYSTEM_PROMPT)
                prompts.submit(change_prompt, [prompts], [prompts], show_progress=True)

            chatbot = gr.Chatbot(elem_classes="control-height", show_copy_button=True)
            query = gr.Textbox(lines=2, label='Input', show_copy_button=True)
            task_history = gr.State([])

            with gr.Row():
                empty_btn = gr.Button("清除历史")
                submit_btn = gr.Button("发送")
                regen_btn = gr.Button("重试")

            submit_btn.click(predict, [query, chatbot, task_history, top_p, temperature, max_tokens], [chatbot], show_progress=True)
            submit_btn.click(reset_user_input, [], [query])
            empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
            regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

app = gr.mount_gradio_app(app, demo, path='/gradio')
