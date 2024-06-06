import time
from transformers import GenerationConfig

def chat(
    messages,
    model,
    tokenizer,
    generate_config=None,
    max_length=512,
    max_new_tokens=256,
):
    if generate_config is None:
        generate_config = GenerationConfig(
            do_sample=False,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            eos_token_id=32001,
        )
    
    if messages[0]["role"] == "system":
        system = messages[0]["content"]
        messages = messages[0:]
    else:
        system = "You are a helpful assistant."
    
    n_token = max_length
    system = "<round_start>system\n{}<round_end>\n".format(system)
    system_token = tokenizer.encode(system, add_special_tokens=False)
    n_token -= len(system_token)

    query = messages[-1]["content"]
    query = "<round_start>user\n{}<round_end>\n<round_start>assistant\n".format(query)
    query_token = tokenizer.encode(query, add_special_tokens=False)
    n_token -= len(query_token)
    
    messages = messages[:-1]
    conversations = []
    for ids in range(len(messages)-1, 0, -2):
        user = messages[ids - 1]["content"]
        assistant = messages[ids]["content"]
        
        round = "<round_start>user\n{}<round_end>\n<round_start>assistant\n{}<round_end>\n".format(user, assistant)
        round_token = tokenizer.encode(round, add_special_tokens=False)

        if n_token - len(round_token) > 0:
            conversations = [round] + conversations
        else:
            break

    prompt = system + "".join(conversations) + query
    prompt_token = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_token.to(model.device)

    response = model.generate(
        generation_config=generate_config,
        **prompt_token
    )
    output_tokens = response[0].cpu().numpy()[prompt_token.input_ids.size()[1]:]
    output_string = tokenizer.decode(output_tokens, skip_special_tokens=True).replace("<round_end>", "")
    return output_string, prompt

# ======================
#       main
# ======================
if __name__ == "__main__":
    query = "Hello!"
    messages = [
        # {"role": "system", "content": "You are an AI assistant, aiming to always upholding high standards of performance and quality."},
        {"role": "user", "content": query}
    ]
    response, input_prompt = chat(messages=messages, model=model, tokenizer=tokenizer)

    print("=" * 25, " User ", "=" * 25,)
    print(query)
    print("=" * 25, " Assistant ", "=" * 25,)
    print(response)