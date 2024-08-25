import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform

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

  def forward(self, X): # X.shape = [n_particle, n_dim_of_theta]

    distances = torch.cdist(X, X, p=2)
    # print(X.shape)
    # print('d', distances.shape, distances)
    dnorm2 = distances ** 2

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()
    
    # print(K_XY.shape, K_XY)
    
    return K_XY
  
# # Let us initialize a reusable instance right away.
# K = RBF()


class SVGD(torch.optim.Adam):
    def __init__(self, params, lr, betas, weight_decay, model_list, train_module):
        super(SVGD, self).__init__(params, lr, betas, weight_decay)
        self.netlist = model_list
        self.K = RBF()
        self.X = params
        self.num_particles = len(model_list)
        self.lr = lr
        self.train_module = train_module
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
                            # print('A', p_)
                            q_A = torch.cat((q_A, p_.cuda()), dim=0)
                        elif "lora_B" in n:
                            # print('B', p_)
                            p_ = p.grad.data.view(1, 1, -1)
                            q_B = torch.cat((q_B, p_.cuda()), dim=0)
                    elif "value" in n:
                        if "lora_A" in n:
                            p_ = p.grad.data.view(1, 1, -1)
                            v_A = torch.cat((v_A, p_.cuda()), dim=0)
                        elif "lora_B" in n:
                            p_ = p.grad.data.view(1, 1, -1)
                            v_B = torch.cat((v_B, p_.cuda()), dim=0)
                            
        # print('q_A_grad', q_A)
        # print('q_A_grad', q_B)
        # print('q_A_grad', v_A)
        # print('q_A_grad', v_B)
            
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
    

    def svgd_kernel(self, theta, h = -1):
        
        
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
    
    def kernel_func(self, q_A, q_B, v_A, v_B):

        q_A.requires_grad = True
        q_B.requires_grad = True
        v_A.requires_grad = True
        v_B.requires_grad = True
        
        kernel_qA = self.K(q_A)
        self.train_module.manual_backward(kernel_qA.sum())
        q_A_grad = q_A.grad
        # print('theta.grad', q_A.grad)
        # exit()
        
        kernel_qB = self.K(q_B)
        self.train_module.manual_backward(kernel_qB.sum())
        q_B_grad = q_B.grad
        
        kernel_vA = self.K(v_A)
        self.train_module.manual_backward(kernel_vA.sum())
        v_A_grad = v_A.grad
        
        kernel_vB = self.K(v_B)
        self.train_module.manual_backward(kernel_vB.sum())
        v_B_grad = v_B.grad
            
        return kernel_qA, kernel_qB, kernel_vA, kernel_vB, q_A_grad, q_B_grad, v_A_grad, v_B_grad
        

    def score_func(self):
        q_A_grad, q_B_grad, v_A_grad, v_B_grad = self.get_grad_allP() #dlog_prob(X)

        self.zero_grad()
        q_A, q_B, v_A, v_B = self.get_learnable_block_allP()
        # q_A.data += torch.rand(q_A.shape).cuda()
        q_A, q_B, v_A, v_B = q_A.clone().detach().requires_grad_(True), q_B.clone().detach().requires_grad_(True), v_A.clone().detach().requires_grad_(True), v_B.clone().detach().requires_grad_(True)
        kernel_qA, kernel_qB, kernel_vA, kernel_vB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK = self.kernel_func(q_A, q_B, v_A, v_B) #self.K(self.X, self.X.detach())
        
        # print(q_A_gradK.shape, q_A_grad.shape, kernel_qA.shape, kernel_qA.detach().matmul(q_A_grad).shape)
        grad_qA = (-kernel_qA.detach().matmul(q_A_grad) + q_A_gradK) / self.num_particles
        grad_qB = (-kernel_qB.detach().matmul(q_B_grad) + q_B_gradK) / self.num_particles
        grad_vA = (-kernel_vA.detach().matmul(v_A_grad) + v_A_gradK) / self.num_particles
        grad_vB = (-kernel_vB.detach().matmul(v_B_grad) + v_B_gradK) / self.num_particles
        
        # print("q_A_grad", q_A_grad)
        # print("kernel_qA", kernel_qA)
        # print("q_A_gradK", q_A_gradK)
        # print("grad_qA", grad_qA)
                
        # exit()

        return grad_qA, grad_qB, grad_vA, grad_vB
        return q_A_grad, q_B_grad, v_A_grad, v_B_grad

    def step_(self):
        q_A_grad, q_B_grad, v_A_grad, v_B_grad = self.score_func()
                
        for net_id in range(self.num_particles):
            for n, p in self.netlist[net_id].named_parameters():
                for layer_id in range(12):
                    if p.requires_grad and str(layer_id) in n:
                        # print(n)
                        if "query" in n:
                            if "lora_A" in n:
                                # print(type(p), type(q_A_grad), type(p.data))
                                # exit()
                                # print('b', p.data)
                                p.data = p.data + self.lr * q_A_grad[layer_id][net_id].view(p.data.shape)
                                # print(self.lr, q_A_grad[layer_id][net_id].view(p.data.shape))
                                # print('a', p.data)
                                # exit()
                            elif "lora_B" in n:
                                # print('b', p.data)
                                p.data = p.data + self.lr * q_B_grad[layer_id][net_id].view(p.data.shape)
                                # print('a', p.data)
                        elif "value" in n:
                            if "lora_A" in n:
                                p.data = p.data + self.lr * v_A_grad[layer_id][net_id].view(p.data.shape)
                            elif "lora_B" in n:
                                p.data = p.data + self.lr * v_B_grad[layer_id][net_id].view(p.data.shape)
