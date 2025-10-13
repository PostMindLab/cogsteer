# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple
import random

import numpy as np
import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm

from llama_detox import LLaMA, ModelArgs, Tokenizer, Transformer
from llama_detox import InnerDetoxHook
from typing import Iterable, List, TypeVar
T = TypeVar('T')
def repeat_interleave(data: Iterable[T], repeats: int) -> Iterable[T]:
    """
    repeat each item in data for repeats times
    """
    for item in data:
        for _ in range(repeats):
            yield item

innerdetox_hook_config = dict(
    type='BaseInnerDetoxHook',
    norm_exp=0.4, #ne
    # neg_sim_exp=0.6, #nse
    neg_sim_exp=0, #nse
    renorm=True,
)
innerdetox_hook = InnerDetoxHook.build(innerdetox_hook_config)
device = "cuda"
# 忽略警告

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    adapter_layer: int,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    quantizer: bool=False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading adapter")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    model_args.adapter_layer = adapter_layer
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    tox_model = Transformer(model_args)
    tox_model.load_state_dict(checkpoint, strict=False)
    tox_model.load_state_dict(adapter_checkpoint["model"], strict=False)

    generator = LLaMA(tox_model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def generate_answers(generator, layer, data_path, save_path):
    k = 25
    index=0
    answers= []
    with open(data_path, 'r', encoding='utf8') as f:
        json_filedata = f.readlines()
    for line in tqdm(json_filedata, position=0, leave=True):
        
        data = json.loads(line)

        prompt = "Please continue the text in English only: " + data['prompt']['text']


        prompts = list(repeat_interleave([prompt], k))
        continuation = generator.generate(prompts, max_gen_len=20, temperature=0.5,
        top_p=0.9, adap_layer=layer, innerdetox_hook=innerdetox_hook)
        continuations = [{'text': cont} for cont in continuation]

        answers.append({
                'index': index,
                'prompt': {"text": prompt},
                'continuation': continuations
            })
        index += 1  # 增加全局索引
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_name = data_path.split('/')[-1].split('.')[0]
    json.dump(answers, open(f'{save_path}/{save_name}_answers.json', 'w'))

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
    
#for name in  ['gpt2-medium', 'gpt2-large', 'gpt2-xl']:
def main():
    seed=42
    set_seed(seed)
    data_dir = '../../data/rtp-test-toxic-2k.jsonl'
    print(data_dir)
    save_dir = "llama-2-7b-result"
    ckpt_dir = "model path"
    tokenizer_path = "tokenizer path"

    for l in range(0, 32):
        print("layer", l) 
        adapter_path = f"../finetune/checkpoint/toxic_llama-2-7b_{l}/checkpoint-5.pth"
        generator = load(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, adapter_path=adapter_path,
                          local_rank=local_rank, world_size=world_size, adapter_layer=l)
        save_path = f"{save_dir}/k_eval/layer_{l}"
        print("save_path:", save_path)
        generate_answers(generator, l, data_dir, save_path)

        # 同步CUDA，确保所有操作完成
        torch.cuda.synchronize()

        # 释放CUDA缓存
        del generator
        torch.cuda.empty_cache()
   


if __name__ == '__main__':
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    fire.Fire(main)
    

    
