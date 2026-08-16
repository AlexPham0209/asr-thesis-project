from transformers import AutoTokenizer


def create_llama3_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llama_chat_template = (
        "{%- for message in messages %}"
        "{%- if loop.index0 == 0 %}{{- bos_token }}{%- endif %}"
        "{%- if message['role'] == 'assistant' %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{%- generation %}"
        "{{- message['content'] | trim + '<|eot_id|>' }}"
        "{%- endgeneration %}"
        "{%- else %}"
        "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{%- generation %}"
        "{%- endgeneration %}"
        "{%- endif %}"
    )

    tokenizer.chat_template = llama_chat_template

    return tokenizer