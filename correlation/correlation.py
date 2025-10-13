import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast
from utils import *
from llama import LLaMA, ModelArgs, Tokenizer, Transformer
from pathlib import Path
import os
import pandas as pd
import torch

def main(dataset, model_type, device):

    if model_type == 'gpt2':
        model_path = 'gpt2'
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        fast_tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    elif model_type == 'gpt2-m':
        model_path = 'gpt2-medium'
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        fast_tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    elif model_type == 'gpt2-l':
        model_path = 'gpt2-large'
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        fast_tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    elif model_type == 'llama2-7b-origin':
        model_path = 'meta-llama model path'
        tokenizer_path = "tokenizer path for llama"
        model, tokenizer = load_origin(model_path, tokenizer_path)
        model = model.to(device)
        tokenizer = tokenizer
        fast_tokenizer = tokenizer.sp_model
    
    print(f"Process {dataset}")
    eye_data = f"normed_efs/{dataset}.json"
    eye_dict = json.load(open(eye_data, 'r'))
    sentences = eye_dict['sents']
    eye_features = eye_dict['eye_fs'] 

    name_ef = {'gd': eye_features[0], 'trt': eye_features[1], 
            'ffd': eye_features[2], 'sfd': eye_features[3], 'gpt': eye_features[4]}
    
    reading_features = get_reading(sentences, model, tokenizer, fast_tokenizer, model_type, device)
    print("num of sentence after reading:", len(reading_features))

    res = []
    initial_other = {"sents": [], "eye_fs": []}
    others_dict = {'gd':initial_other, 'trt': initial_other, 'ffd': initial_other, 'sfd': initial_other, 'gpt': initial_other}
    for eye_key, eye_feat in name_ef.items():
        print('-'*14,eye_key, '-'*14)
        num_layers = reading_features[0].shape[0]
        for cur_layer in range(num_layers):
            cur_layer_feats = [sentence[cur_layer,:,:] for sentence in reading_features] # list: num_sent * torch.Size([num_subtokens, emb_size])
            cor_dict, others = F_analysis_pca(eye_feat, cur_layer_feats)


            cor_dict["layer"] = cur_layer
            cor_dict["eye"] = eye_key
            res.append(cor_dict)

            for i in others:
                if sentences[i] not in others_dict[eye_key]['sents']:
                    others_dict[eye_key]['sents'].append(sentences[i])
                    others_dict[eye_key]['eye_fs'].append(name_ef[eye_key][i])


    df = pd.DataFrame(res)
    os.makedirs(f'results/{model_type}', exist_ok=True)
    df.to_csv(f'results/{model_type}/correlation_pca_{dataset}.csv', index=False)

def load_origin(ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int =0,
    world_size: int =1 ,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
)-> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = "origin"
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    return model, tokenizer

if __name__ == '__main__':
    datasets = ['Zuco_NR', 'Zuco_TSR', 'Provo', 'Geco']
    model_type = "gpt2" # gpt2-l, gpt2-m, gpt2, llama2-7b-origin
    device = "cuda"
    for dataset in datasets:
        main(dataset, model_type, device)