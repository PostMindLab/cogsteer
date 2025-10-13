# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama_detox.model import Transformer
from llama_detox.tokenizer import Tokenizer
import sys


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        adap_layer: int = -1,
        innerdetox_hook = None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        if adap_layer != 32:
            adap = f"layers.{adap_layer}.attention.before_mergehead"
        elif adap_layer == 32:
            adap = [f"layers.{l}.attention.before_mergehead" for l in range(1, 32)]

        # tox_tokens = tokens.clone()
        for cur_pos in range(start_pos, total_len):

            if adap_layer != -1 and innerdetox_hook is not None:
                # print("adap", adap)
                innerdetox_hook.register_hooks(self.model, innerdetox_hook.get_attn_output, adap_layer, adap, None)
                tox_logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, adap_layer)
                attn_output = innerdetox_hook.attn_output.copy()
                innerdetox_hook.remove_hooks()
                innerdetox_hook.remove_attn_output()

                innerdetox_hook.register_hooks(self.model, innerdetox_hook.reversal_func, adap_layer, adap, attn_output)
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, "origin")
                innerdetox_hook.remove_hooks()
                innerdetox_hook.remove_attn_output()
            else:
                # default
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            if adap_layer == 31 or adap_layer == "full":
                # last layer or full adapter use cd to detox
                logits = logits - 0.1* tox_logits


            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
