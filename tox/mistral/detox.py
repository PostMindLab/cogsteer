from models.innerdetox_hook import InnerDetoxHook
from transformers import AutoTokenizer
import adapters
# from adapters import MistralAdapterModel
# from models.adapter_init import init
from models.modeling_mistral_innerdetox import MistralForCausalLM
# from transformers import MistralForCausalLM
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import repeat_interleave
import json
from tqdm import tqdm
import random
import torch
import numpy as np


model_path = "mistralai/Mistral-7B-v0.3"
adapter_name = "Mistral-7B"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

innerdetox_hook_config = dict(
    type='BaseInnerDetoxHook',
    norm_exp=0.4, #ne
    # neg_sim_exp=0.6, #nse
    neg_sim_exp=0, #nse
    renorm=True,
)
innerdetox_hook = InnerDetoxHook.build(innerdetox_hook_config)

def batch_generate(model, prompts):
    # 批量编码句子
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, max_length=None, truncation=True)

    # 计算最长句子的长度，并添加额外的 20 个 token
    max_length = encodings['input_ids'].shape[1] + 20

    # 批量生成续写
    continuations = model.generate(
        input_ids=encodings["input_ids"].to(device),
        attention_mask=encodings["attention_mask"].to(device),
        pad_token_id=tokenizer.pad_token_id,
        temperature=1,
        top_p=0.9,
        do_sample=True,
        max_length=max_length,
        num_return_sequences=1,
        innerdetox_hook=innerdetox_hook,
    )

    # 解码生成的文本并添加到列表中
    continuation = []
    for conti in continuations:
        text = tokenizer.decode(conti, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        text = text[:text.find(tokenizer.eos_token)]
        text = text.replace("<s>", "").replace("</s>", "")
        continuation.append(text)
    # print(continuation)
    return continuation

def generate_answers(model, layer, data_path, save_path):
    k = 25
    index=0
    answers= []
    with open(data_path, 'r', encoding='utf8') as f:
        json_filedata = f.readlines()
    for line in tqdm(json_filedata, position=0, leave=True):
        
        data = json.loads(line)

        prompt = data['prompt']['text']

        prompts = list(repeat_interleave([prompt], k))
        continuation = batch_generate(model, prompts)
        continuations = [{'text': cont} for cont in continuation]

        answers.append({
                'index': index,
                'prompt': {"text": prompt},
                'continuations': continuations
            })
        index += 1  # 增加全局索引
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_name = data_path.split('/')[-1].split('.')[0]
    json.dump(answers, open(f'{save_path}/{save_name}_answers.json', 'w'))

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed=42
    set_seed(seed)
    # l = 11
    print(adapter_name)
    for l in range(0, 32):
        # l="full"
        print("layer", l)

        ori_model = MistralForCausalLM.from_pretrained(model_path, attn_implementation="sdpa")
        ori_model.to(device)
        tox_model = MistralForCausalLM.from_pretrained(model_path, attn_implementation="sdpa")
        adapters.init(tox_model)
        tox_model.load_adapter("./weights/toxic_{}_layer_{}".format(adapter_name, l))
        tox_model.set_active_adapters("toxic_{}_{}".format(adapter_name, l))
        tox_model.to(device)
        ori_model.set_tox_model(tox_model, l)

        data_dir = '../data/rtp-test-toxic-2k.jsonl'
        generate_answers(ori_model, str(l), data_dir,f"k_eval/layer_{l}")

        torch.cuda.synchronize()

        # 释放CUDA缓存
        del ori_model
        del tox_model
        torch.cuda.empty_cache()



