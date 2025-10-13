from functools import partial
from mmengine import Registry
import torch
import torch.nn.functional as F

InnerDetoxHook = Registry('innerdetox_hook')


@InnerDetoxHook.register_module()
class BaseInnerDetoxHook():
    def __init__(self, norm_exp=0, neg_sim_exp=0, renorm=False):
        self.mem = dict()
        self.hook_handles = dict()
        self.norm_exp = norm_exp
        self.neg_sim_exp = neg_sim_exp
        self.renorm = renorm
        self.attn_output = dict()
    
    def get_attn_output(self, module, input, output, module_name=None, attn_output=None):
        self.attn_output[module_name] = output.detach()

    def reversal_func(self, module, input, output, module_name=None, attn_output=None):
        if self.mem.get(module_name, None) is None:
            self.mem[module_name] = dict()
        if attn_output is not None and attn_output.get(module_name, None) is not None:
            self.mem[module_name]['delta']= ( attn_output[module_name][:,:,-1:,:] - output[:,:,-1:,:] ).detach()
        else:
            print("reversal_func: ", attn_output)
            raise ValueError("attn_output is None")

        v = output
        delta = self.mem[module_name]['delta']
        if self.renorm:
            v_norm = v[:,:,-1:,:].norm(dim=(1,3), keepdim=True)

        norm_scale = 1
        if self.norm_exp > 0:
            norm_scale = (1 + delta.norm(dim=-1, keepdim=True)) ** self.norm_exp

        v[:,:,-1:,:] = v[:,:,-1:,:] - norm_scale  * delta

        if self.renorm:
            new_v_norm = v[:,:,-1:,:].norm(dim=(1,3), keepdim=True)
            v[:,:,-1:,:] = v[:,:,-1:,:] * (v_norm / new_v_norm)
        return v

    def register_hooks(self, model, hook, layer, adap, attn_output=None):
        ctn=0
        for n, m in model.named_modules():
            if self.module_match_fn(n):
                if layer == 32:
                    if n in adap:
                        handle = m.register_forward_hook(partial(hook, module_name=n, attn_output=attn_output))
                        self.hook_handles[n] = handle
                        ctn+=1
                else:
                    if n == adap:
                        handle = m.register_forward_hook(partial(hook, module_name=n, attn_output=attn_output))
                        self.hook_handles[n] = handle
                        ctn+=1
        if ctn == 0:
            raise ValueError("No layer found in the model")

    def remove_hooks(self):
        for n in list(self.hook_handles.keys()):
            self.hook_handles[n].remove()
            self.hook_handles.pop(n)
    
    def remove_attn_output(self):
        for n in list(self.attn_output.keys()):
            self.attn_output.pop(n)
    
    