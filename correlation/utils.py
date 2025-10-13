import math, re
import json
import torch
from transformers import AutoTokenizer, BertLMHeadModel, LlamaForCausalLM, LlamaTokenizerFast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast
import math, os
import numpy as np

#from rwkv_tools import generate, get_tokenizer
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr
#from rwkv_tools import generate_at_once, get_tokenizer
import string
from tqdm import tqdm
import sys

def Embeding_reduce(log_n_tokens, E):
    """
    log_n_tokens: list in shape of sentence_len, means how many tokens dose one raw word has been tokenized into
    E: torch.Size([num_subtoken, emb_size])
    return
    E: torch.Size([num_raw_words, emb_size])
    """
    start = 0
    new_E = []
    for n in log_n_tokens:
        max_val, _ = torch.max(E[start:start+n, :], dim=0)
        new_E.append(max_val)
        start += n

    return torch.stack(new_E).detach()



def Hidden_reduce(log_n_tokens, H):
    """
    log_n_tokens: list in shape of sentence_len, means how many tokens dose one raw word has been tokenized into
    H: num_layer * torch.Size([num_subtoken, emb_size])
    return:
    H: tensor.shape= (num_layers, num_subtoken, emb_size)
    """
    new_H = []
    for layer in H:
        new_H.append(Embeding_reduce(log_n_tokens, layer))

    new_H = torch.stack(new_H)

    return new_H


def get_len_sub(sentence, tokenizer):
    encoding  = tokenizer.encode_plus(sentence, add_special_tokens=False, return_offsets_mapping=True)
    # tokens列表
    tokens = encoding["input_ids"]
    
    offsets = encoding["offset_mapping"]

    word_token_counts = [0] * len(sentence.split(' '))
    current_word_idx = 0

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        token = sentence[start:end]
        current_word = sentence.split()[current_word_idx]
        if token in current_word:
            word_token_counts[current_word_idx] += 1
        else:
            current_word_idx += 1
            word_token_counts[current_word_idx] += 1
    return word_token_counts


def get_len_sub_Llama(sentence, tokenizer):
    encoding  = tokenizer.encode_plus(sentence, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoding["input_ids"]

    offsets = encoding["offset_mapping"]

    word_token_counts = [0] * len(sentence.split())
    current_word_idx = 0

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        token = sentence[start:end]
        current_word = sentence.split()[current_word_idx]
        if token in current_word:
            word_token_counts[current_word_idx] += 1
        else:
            current_word_idx += 1
            word_token_counts[current_word_idx] += 1
    return word_token_counts

def get_len_sub_llama_origin(encoding, tokenizer, word_len):
    word_token_counts = [0] * word_len
    current_word_idx = 0
    
    for token_id in encoding:
        token = tokenizer.id_to_piece(token_id)
        if token.startswith("▁"):
            if current_word_idx < word_len:
                current_word_idx += 1
        if current_word_idx <= len(word_token_counts):
            word_token_counts[current_word_idx - 1] += 1

    assert current_word_idx == word_len, 'Token count does not match word count'
    return word_token_counts

def reading_sent(model, tokenizer, fast_tokenizer, sent, model_type, device):
    with torch.no_grad():  # no tracking history
        if 'gpt2' in model_type:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

            log_n_tokens = get_len_sub(sent, fast_tokenizer)
            H = Hidden_reduce(log_n_tokens, [torch.squeeze(layer) for layer in outputs.hidden_states[0:]])

            return H

        elif 'llama2-7b-origin' == model_type:
            inputs = fast_tokenizer.encode(sent)
            word_len = len(sent.split())
            log_n_tokens = get_len_sub_llama_origin(inputs, fast_tokenizer, word_len)

            inputs = torch.tensor(inputs).unsqueeze(0).to(device)
            h = model(inputs)

            # get rid of start sign
            H = Hidden_reduce(log_n_tokens, [torch.squeeze(layer) for layer in h])
            return H

        

def get_reading(sents, model, tokenizer, fast_tokenizer, model_type, device):
    """
    :param sents: list of str, [sent1, sent2 ... sent_n]
    :param model: original model
    :param model_type: type of original model
    :return: list of hidden state, hidden_state.shape (num_layer, num_subtokens, emb_size)
    """
    reading_features = []
    for sent in tqdm(sents):
        h = reading_sent(model, tokenizer, fast_tokenizer, sent, model_type, device)
        reading_features.append(h)

    return reading_features


def cal_relation(E, R):
    #calculate the corelation of both raw signals and loged signals with eye tracking features
    data = {'eye_f':E, 'read_f':R}
    df = pd.DataFrame(data)
    pc, pp, kc, kp, sc, sp = calcaulate_correlatuons(df)

    R_log = [math.log10(x) if x > 0 else x for x in R]
    data_log = {'eye_f':E, 'read_f':R_log}
    df_log = pd.DataFrame(data_log)
    pcl, ppl, kcl, kpl, scl, spl = calcaulate_correlatuons(df_log)

    return {"pearson_cor":pc, "pearson_p":pp, "kendall_cor":kc, "kendall_p":kp, "spearman_cor":sc, "spearman_p":sp, 
            "pearson_cor_log":pcl, "pearson_p_log":ppl, "kendall_cor_log":kcl, "kendall_p_log":kpl, "spearman_cor_log":scl, "spearman_p_log":spl}

def calcaulate_correlatuons(df):
    pearson_corr, pearson_p_value = pearsonr(df['eye_f'], df['read_f'])

    kendall_corr, kendall_p_value = kendalltau(df['eye_f'], df['read_f'])

    spearman_corr, spearman_p_value = spearmanr(df['eye_f'], df['read_f'])

    return pearson_corr,  pearson_p_value, kendall_corr, kendall_p_value, spearman_corr, spearman_p_value


def F_analysis_pca(eye_feat, cur_layer_feats):
    pca = PCA(n_components=1)
    
    others = []
    efs, layer_feats= [], []
    for i, (ef, layer_feat) in enumerate(zip(eye_feat, cur_layer_feats)):
        if len(ef) != len(layer_feat): # token数
            others.append(i)
        else:
            efs.extend(ef)
            layer_feat = layer_feat.tolist()
            layer_feats.extend(layer_feat) 

    # all token in pca 
    pca.fit(layer_feats)
    layer_feats = pca.transform(layer_feats).flatten().tolist() 

    return cal_relation(efs, layer_feats), others