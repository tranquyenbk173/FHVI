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

    if X.shape[0] == 0:
        return 0

    distances = torch.cdist(X, X, p=2)
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
        
    return K_XY
  

class SVGD(torch.optim.Adam):
    def __init__(self, param, lr, betas, weight_decay, num_particles, train_module, net):
        super(SVGD, self).__init__(param, lr, betas, weight_decay)
        self.K = RBF()
        self.net = net
        self.num_particles = num_particles
        self.lr = lr
        self.train_module = train_module
        
        print(self.net)
            
        # self.based_optim = torch.optim.Adam(
        #         params,
        #         lr=lr,
        #         betas=betas,
        #         weight_decay=weight_decay,
        #     )
        
    def get_learnable_block(self): #for LoRA        
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        # print('Get learnable params')
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                # print(n)
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                q_A = torch.cat((q_A, tq_A), dim=0)
                                tq_A = torch.empty(0).cuda()
                                i_qA = 0
                    elif "w_b" in n:
                        if i_qB < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tq_B = torch.cat((tq_B, p_), dim=1)
                            i_qB += 1
                            if i_qB == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                q_B = torch.cat((q_B, tq_B), dim=0)
                                tq_B = torch.empty(0).cuda()
                                i_qB = 0
                elif "proj_v" in n:
                    if "w_a" in n:
                        if i_vA < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tv_A = torch.cat((tv_A, p_), dim=1)
                            i_vA += 1
                            if i_vA == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                v_A = torch.cat((v_A, tv_A), dim=0)
                                tv_A = torch.empty(0).cuda()
                                i_vA = 0
                    elif "w_b" in n:
                        if i_vB < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tv_B = torch.cat((tv_B, p_), dim=1)
                            i_vB += 1
                            if i_vB == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                v_B = torch.cat((v_B, tv_B), dim=0)
                                tv_B = torch.empty(0).cuda()
                                i_vB = 0
                            
        return q_A, q_B, v_A, v_B
    
    def get_learnable_block_allP(self):
        q_A, q_B, v_A, v_B = self.get_learnable_block()
        return q_A, q_B, v_A, v_B
    
    def get_grad(self): #for LoRA
        
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                q_A = torch.cat((q_A, tq_A), dim=0)
                                tq_A = torch.empty(0).cuda()
                                i_qA = 0
                    elif "w_b" in n:
                        if i_qB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_B = torch.cat((tq_B, p_), dim=1)
                            i_qB += 1
                            if i_qB == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                q_B = torch.cat((q_B, tq_B), dim=0)
                                tq_B = torch.empty(0).cuda()
                                i_qB = 0
                elif "proj_v" in n:
                    if "w_a" in n:
                        if i_vA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_A = torch.cat((tv_A, p_), dim=1)
                            i_vA += 1
                            if i_vA == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                v_A = torch.cat((v_A, tv_A), dim=0)
                                tv_A = torch.empty(0).cuda()
                                i_vA = 0
                    elif "w_b" in n:
                        if i_vB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_B = torch.cat((tv_B, p_), dim=1)
                            i_vB += 1
                            if i_vB == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                v_B = torch.cat((v_B, tv_B), dim=0)
                                tv_B = torch.empty(0).cuda()
                                i_vB = 0
                            
        print('q_A_grad', q_A.shape, q_A)
        print('q_B_grad', q_B.shape, q_B)
        print('v_A_grad', v_A.shape, v_A)
        print('v_B_grad', v_B.shape, v_B)
                    
        return q_A, q_B, v_A, v_B
    
    def get_grad_allP(self):
        q_A, q_B, v_A, v_B = self.get_grad() 
        return q_A, q_B, v_A, v_B
    
    
    def kernel_func(self, q_A, q_B, v_A, v_B):

        q_A.requires_grad = True
        q_B.requires_grad = True
        v_A.requires_grad = True
        v_B.requires_grad = True
        
        kernel_qA = self.K(q_A)
        self.train_module.manual_backward(kernel_qA.sum())
        q_A_grad = q_A.grad
        
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
        q_A_grad, q_B_grad, v_A_grad, v_B_grad = self.get_grad_allP() #dlog_prob(X)'
        
        try:
            cls_grad_w = self.net.lora_vit.fc.weight.grad.data
            cls_grad_b = self.net.lora_vit.fc.bias.grad.data
        except:
            cls_grad_w = self.net.fc.weight.grad.data
            cls_grad_b = self.net.fc.bias.grad.data
            
        print('gradd_cls', cls_grad_w.sum(), cls_grad_w)
        
        self.zero_grad()
        q_A, q_B, v_A, v_B = self.get_learnable_block_allP()
        q_A, q_B, v_A, v_B = q_A.clone().detach().requires_grad_(True), q_B.clone().detach().requires_grad_(True), v_A.clone().detach().requires_grad_(True), v_B.clone().detach().requires_grad_(True)
        if q_A.shape[0] > 0:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK = self.kernel_func(q_A, q_B, v_A, v_B) #self.K(self.X, self.X.detach())
        else:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK = torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), 0, 0, 0, 0
        
        grad_qA = (-kernel_qA.detach().matmul(q_A_grad) + q_A_gradK) / self.num_particles
        print(grad_qA.sum(), grad_qA + q_A_grad)
        # print(grad_qA)
        # print(q_A_grad)
        
        print("*"*50)
        # exit()
        
        grad_qB = (-kernel_qB.detach().matmul(q_B_grad) + q_B_gradK) / self.num_particles
        print(grad_qB.sum(), grad_qB + q_B_grad)
        # print(kernel_qA)
        # print(q_A_gradK)
        # exit()
        grad_vA = (-kernel_vA.detach().matmul(v_A_grad) + v_A_gradK) / self.num_particles
        print(grad_vA.sum(), grad_vA + v_A_grad)
        grad_vB = (-kernel_vB.detach().matmul(v_B_grad) + v_B_gradK) / self.num_particles
        print(grad_vB.sum(), grad_vB + v_B_grad)
        
        # exit()
        
        return grad_qA, grad_qB, grad_vA, grad_vB, cls_grad_w, cls_grad_b

    def step_(self):
        
        
        for n, p in self.net.named_parameters():
            if p.requires_grad == True: 
                print(n)
                p.data = p.data - self.lr * p.grad.data
                
        
        print('update done')
        return
        
        
        
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, cls_grad_w, cls_grad_b = self.score_func()
        
        for net_id in range(self.num_particles):        
            for n, p in self.net.named_parameters():
                for layer_id in range(12):
                    if p.requires_grad and f'blocks.{str(layer_id)}' in n:
                        if "proj_q" in n:
                            if f"w_a.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr *self.num_particles* q_A_grad[layer_id][net_id].view(p.data.shape)
                            elif f"w_b.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr *self.num_particles* q_B_grad[layer_id][net_id].view(p.data.shape)
                        elif "proj_v" in n:
                            if f"w_a.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr *self.num_particles* v_A_grad[layer_id][net_id].view(p.data.shape)
                            elif f"w_b.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr *self.num_particles* v_B_grad[layer_id][net_id].view(p.data.shape)
        
        try:                        
            temp_w = self.net.lora_vit.fc.weight.data
            temp_b = self.net.lora_vit.fc.bias.data
        except:
            temp_w = self.net.fc.weight.data
            temp_b = self.net.fc.bias.data
        
        try:
            self.net.lora_vit.fc.weight.data =  temp_w - self.lr/self.num_particles * cls_grad_w
            self.net.lora_vit.fc.bias.data =  temp_b - self.lr/self.num_particles * cls_grad_b
        except:
            self.net.fc.weight.data =  temp_w - self.lr/self.num_particles * cls_grad_w
            self.net.fc.bias.data =  temp_b - self.lr/self.num_particles * cls_grad_b
                                
        
