import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim

def block_expansion(ckpt, split, original_layers):

    layer_cnt = 0
    selected_layers = []
    output = {}

    for i in range(original_layers):
        for k in ckpt:
            if ('layer.' + str(i) + '.') in k:
                output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('layer.' + str(i) + '.') in k:
                    if 'attention.output' in k or str(i)+'.output' in k:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = torch.zeros_like(ckpt[k])
                        selected_layers.append(layer_cnt)
                    else:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
            layer_cnt += 1

    for k in ckpt:
        if not 'layer' in k:
            output[k] = ckpt[k]
        elif k == "vit.layernorm.weight" or k == "vit.layernorm.bias" or k == "dinov2.layernorm.bias" or k == "dinov2.layernorm.weight":
            output[k] = ckpt[k]
    
    selected_layers = list(set(selected_layers))

    return output, selected_layers


class RBF(torch.nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()
    
    # print('k shape', K_XY.shape)

    return K_XY
  
# # Let us initialize a reusable instance right away.
# K = RBF()


class SVGD(torch.optim.Adam):
    def __init__(self, params, lr, betas, weight_decay, model_list):
        super(SVGD, self).__init__(params, lr, betas, weight_decay)
        self.netlist = model_list
        self.K = RBF()
        self.X = params
        self.num_particles = len(model_list)
        self.lr = lr
        # self.based_optim = torch.optim.Adam(
        #         params,
        #         lr=lr,
        #         betas=betas,
        #         weight_decay=weight_decay,
        #     )
        
    def get_learnable_block(self, net_id): #for LoRA
        if True: 
            q_A = torch.empty(0).cuda()
            q_B = torch.empty(0).cuda()
            v_A = torch.empty(0).cuda()
            v_B = torch.empty(0).cuda()
            for n, p in self.netlist[net_id].named_parameters():
                if p.requires_grad:
                    if "query" in n:
                        if "lora_A" in n:
                            p_ = p.view(1, 1, -1)
                            q_A = torch.cat((q_A, p_.cuda()), dim=0)
                        elif "lora_B" in n:
                            p_ = p.view(1, 1, -1)
                            q_B = torch.cat((q_B, p_.cuda()), dim=0)
                    elif "value" in n:
                        if "lora_A" in n:
                            p_ = p.view(1, 1, -1)
                            v_A = torch.cat((v_A, p_.cuda()), dim=0)
                        elif "lora_B" in n:
                            p_ = p.view(1, 1, -1)
                            v_B = torch.cat((v_B, p_.cuda()), dim=0)
            
        return q_A, q_B, v_A, v_B
    
    def get_learnable_block_allP(self):
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        for j in range(self.num_particles):
            q_Aj, q_Bj, v_Aj, v_Bj = self.get_learnable_block(j)
            q_A = torch.cat((q_A, q_Aj.cuda()), dim=1)
            q_B = torch.cat((q_B, q_Bj.cuda()), dim=1)
            v_A = torch.cat((v_A, v_Aj.cuda()), dim=1)
            v_B = torch.cat((v_B, v_Bj.cuda()), dim=1)
            
        return q_A, q_B, v_A, v_B
    
    def get_grad(self, net_id): #for LoRA
        if True:
            q_A = torch.empty(0).cuda()
            q_B = torch.empty(0).cuda()
            v_A = torch.empty(0).cuda()
            v_B = torch.empty(0).cuda()
            for n, p in self.netlist[net_id].named_parameters():
                if p.requires_grad:
                    if "query" in n:
                        if "lora_A" in n:
                            p_ = p.grad.data.view(1, 1, -1)
                            q_A = torch.cat((q_A, p_.cuda()), dim=0)
                        elif "lora_B" in n:
                            p_ = p.grad.data.view(1, 1, -1)
                            q_B = torch.cat((q_B, p_.cuda()), dim=0)
                    elif "value" in n:
                        if "lora_A" in n:
                            p_ = p.grad.data.view(1, 1, -1)
                            v_A = torch.cat((v_A, p_.cuda()), dim=0)
                        elif "lora_B" in n:
                            p_ = p.grad.data.view(1, 1, -1)
                            v_B = torch.cat((v_B, p_.cuda()), dim=0)
            
        return q_A, q_B, v_A, v_B
    
    def get_grad_allP(self):
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        for j in range(self.num_particles):
            q_Aj, q_Bj, v_Aj, v_Bj = self.get_grad(j)
            q_A = torch.cat((q_A, q_Aj.cuda()), dim=1)
            q_B = torch.cat((q_B, q_Bj.cuda()), dim=1)
            v_A = torch.cat((v_A, v_Aj.cuda()), dim=1)
            v_B = torch.cat((v_B, v_Bj.cuda()), dim=1)
            
        return q_A, q_B, v_A, v_B

    def kernel_func(self):
        q_A, q_B, v_A, v_B = self.get_learnable_block_allP()
        
        kernel_func = 0
        for i in range(12):
            kernel_func += self.K(q_A[i], q_A[i])
            kernel_func += self.K(q_B[i], q_B[i])
            kernel_func += self.K(v_A[i], v_A[i])
            kernel_func += self.K(v_B[i], v_B[i])
            
        # print(kernel_func.shape, kernel_func)
        return kernel_func
        

    def score_func(self):
        q_A_grad, q_B_grad, v_A_grad, v_B_grad = self.get_grad_allP() #dlog_prob(X)

        self.zero_grad()
        K_XX = self.kernel_func() #self.K(self.X, self.X.detach())
        K_XX.sum().backward()
        q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK = self.get_grad_allP() #dK

        def phi(grad_logP, grad_K):
            return (K_XX.detach().matmul(grad_logP) + grad_K) / self.num_particles
            
        q_A_grad, q_B_grad, v_A_grad, v_B_grad = phi(q_A_grad, q_A_gradK), phi(q_B_grad, q_B_gradK), phi(v_A_grad, v_A_gradK), phi(v_B_grad, v_B_gradK)

        return -q_A_grad, -q_B_grad, -v_A_grad, -v_B_grad

    def step_(self):
        # print('Zooooo')
        q_A_grad, q_B_grad, v_A_grad, v_B_grad = self.score_func()
        
        for net_id in range(self.num_particles):
            layer_id =  0
            for n, p in self.netlist[net_id].named_parameters():
                if p.requires_grad and str(layer_id) in n:
                    if "query" in n:
                        if "lora_A" in n:
                            p = p - self.lr * q_A_grad[layer_id][net_id].view(p.shape)
                        elif "lora_B" in n:
                            p = p - self.lr * q_B_grad[layer_id][net_id].view(p.shape)
                    elif "value" in n:
                        if "lora_A" in n:
                            p = p - self.lr * v_A_grad[layer_id][net_id].view(p.shape)
                        elif "lora_B" in n:
                            p = p - self.lr * v_B_grad[layer_id][net_id].view(p.shape)
