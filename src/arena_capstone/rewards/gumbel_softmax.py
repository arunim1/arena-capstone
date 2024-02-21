import torch
import torch.nn as nn
import torch.nn.functional as F

def 



class GumbelSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, probs):
        # probs : [stuff, d_vocab]
        noise = torch.rand_like(probs[..., 0])  # noise: (stuff,)
        acc_probs = probs.cumsum(dim=-1) # acc_probs: (stuff, d_vocab)
        index = (acc_probs < noise.unsqueeze(-1)).sum(dim=-1) # index is i where acc_probs[i - 1] < noise[i] < acc_probs[i]
        ctx.save_for_backward(index, probs)
        return F.one_hot(index, num_classes=probs.size(-1), dtype=probs.dtype)

        # sample = torch.zeros_like(probs).uniform_(0, 1)
        # sample = sample - torch.log(-torch.log(sample))
        # ctx.save_for_backward(sample, probs)
        # return torch.argmax(probs + sample, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        # sample, probs = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input.scatter_(-1, torch.argmax(probs + sample, dim=-1, keepdim=True), 1)
        # return grad_input
        index, probs = ctx.saved_tensors
        back = torch.zeros_like(probs)





class Hmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        # probs : [stuff, d_vocab]
        noise = torch.rand_like(probs[..., 0])  # noise: (stuff,)
        acc_probs = probs.cumsum(dim=-1) # acc_probs: (stuff, d_vocab)
        index = (acc_probs < noise.unsqueeze(-1)).sum(dim=-1) # index is i where acc_probs[i - 1] < noise[i] < acc_probs[i]
        ctx.save_for_backward(index, probs)
        return F.one_hot(index, num_classes=probs.size(-1), dtype=probs.dtype)
        # sample = torch.zeros_like(probs).uniform_(0, 1)
        # sample = sample - torch.log(-torch.log(sample))
        # ctx.save_for_backward(sample, probs)
        # return torch.argmax(probs + sample, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        # sample, probs = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input.scatter_(-1, torch.argmax(probs + sample, dim=-1, keepdim=True), 1)
        # return grad_input
        index, probs = ctx.saved_tensors
        return grad_output * probs