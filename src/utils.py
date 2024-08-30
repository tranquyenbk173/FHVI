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
        cls_w = torch.empty(0).cuda()
        cls_b = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        tcls_w = torch.empty(0).cuda()
        tcls_b = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        i_cls_w = 0
        i_cls_b = 0
        # print('Get learnable params')
        for n, p in self.net.named_parameters():
            if p.requires_grad:
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
                elif "fc" in n:
                    if "weight" in n:
                        if i_cls_w < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tcls_w = torch.cat((tcls_w, p_), dim=1)
                            i_cls_w += 1
                            if i_cls_w == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                cls_w = torch.cat((cls_w, tcls_w), dim=0)
                                tcls_w = torch.empty(0).cuda()
                                i_cls_w = 0
                    elif "bias" in n:
                        if i_cls_b < self.num_particles:
                            p_ = p.data.view(1, 1, -1)
                            tcls_b = torch.cat((tcls_b, p_), dim=1)
                            i_cls_b += 1
                            if i_cls_b == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                cls_b = torch.cat((cls_b, tcls_b), dim=0)
                                tcls_b = torch.empty(0).cuda()
                                i_cls_b = 0
                            
        return q_A, q_B, v_A, v_B, cls_w, cls_b
    
    def get_grad1(self): #for LoRA
        
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        cls_w = torch.empty(0).cuda()
        cls_b = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        tcls_w = torch.empty(0).cuda()
        tcls_b = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        i_cls_w = 0
        i_cls_b = 0
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
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
                                
                elif 'fc' in n:
                    if 'weight' in n:
                        if i_cls_w < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_w = torch.cat((tcls_w, p_), dim=1)
                            i_cls_w += 1
                            if i_cls_w == self.num_particles:   
                                cls_w = torch.cat((cls_w, tcls_w), dim=0)
                                tcls_w = torch.empty(0).cuda()
                                i_cls_w = 0
                    elif 'bias' in n:
                        if i_cls_b < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_b = torch.cat((tcls_b, p_), dim=1)
                            i_cls_b += 1
                            if i_cls_b == self.num_particles:   
                                cls_b = torch.cat((cls_b, tcls_b), dim=0)
                                tcls_b = torch.empty(0).cuda()
                                i_cls_b = 0
                    
                            
        # print('q_A_grad', q_A.shape, q_A)
        # print('q_B_grad', q_B.shape, q_B)
        # print('v_A_grad', v_A.shape, v_A)
        # print('v_B_grad', v_B.shape, v_B)
        # print('cls_w', cls_w.shape)
        # print('cls_b', cls_b.shape)
        # exit()
                    
        return q_A, q_B, v_A, v_B, cls_w, cls_b
    
    
    def get_grad(self):
        q_A_grad = torch.empty(0).cuda()
        q_B_grad = torch.empty(0).cuda()
        v_A_grad = torch.empty(0).cuda()
        v_B_grad = torch.empty(0).cuda()
                
        for n, p in self.net.named_parameters():
            for layer_id in range(12):
                if p.requires_grad and f'blocks.{str(layer_id)}' in n:
                    if "proj_q" in n:
                        for net_id in range(self.num_particles):
                            if f"w_a.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr * q_A_grad[layer_id][net_id].view(p.data.shape)
                            elif f"w_b.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr * q_B_grad[layer_id][net_id].view(p.data.shape)
                    elif "proj_v" in n:
                        for net_id in range(self.num_particles):
                            if f"w_a.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr * v_A_grad[layer_id][net_id].view(p.data.shape)
                            elif f"w_b.{net_id}" in n:
                                temp_w = p.data
                                p.data = temp_w + self.lr * v_B_grad[layer_id][net_id].view(p.data.shape)

    
    
    def kernel_func(self, q_A, q_B, v_A, v_B, clsW, clsB):

        q_A.requires_grad = True
        q_B.requires_grad = True
        v_A.requires_grad = True
        v_B.requires_grad = True
        clsW.requires_grad = True
        clsB.requires_grad = True
        
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
        
        kernel_clsW = self.K(clsW)
        self.train_module.manual_backward(kernel_clsW.sum())
        clsW_gradK = clsW.grad
        
        kernel_clsB = self.K(clsB)
        self.train_module.manual_backward(kernel_clsB.sum())
        clsB_gradK = clsB.grad
        
        return kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_gradK, clsB_gradK
        

    def score_func(self):
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_grad, clsB_grad = self.get_grad1() #dlog_prob(X)'
        
        # try:
        #     cls_grad_w = self.net.lora_vit.fc.weight.grad.data
        #     cls_grad_b = self.net.lora_vit.fc.bias.grad.data
        # except:
        #     cls_grad_w = self.net.fc.weight.grad.data
        #     cls_grad_b = self.net.fc.bias.grad.data
            
        # print('gradd_cls', cls_grad_w.sum(), cls_grad_w)
        
        self.zero_grad()
        q_A, q_B, v_A, v_B, clsW, clsB = self.get_learnable_block()
        q_A, q_B, v_A, v_B, clsW, clsB = q_A.clone().detach().requires_grad_(True), q_B.clone().detach().requires_grad_(True), v_A.clone().detach().requires_grad_(True), v_B.clone().detach().requires_grad_(True),  clsW.clone().detach().requires_grad_(True), clsB.clone().detach().requires_grad_(True)
        
        if q_A.shape[0] > 0:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK = self.kernel_func(q_A, q_B, v_A, v_B, clsW, clsB) #self.K(self.X, self.X.detach())
        else:
            kernel_qA, kernel_qB, kernel_vA, kernel_vB, kernel_clsW, kernel_clsB, q_A_gradK, q_B_gradK, v_A_gradK, v_B_gradK, clsW_gradK, clsB_gradK = torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), torch.ones(size=(q_A_grad.shape[0], q_A_grad.shape[0])).cuda(), 0, 0, 0, 0, 0, 0
        
        grad_qA = (-kernel_qA.detach().matmul(q_A_grad) + q_A_gradK) / self.num_particles
        
        grad_qB = (-kernel_qB.detach().matmul(q_B_grad) + q_B_gradK) / self.num_particles

        grad_vA = (-kernel_vA.detach().matmul(v_A_grad) + v_A_gradK) / self.num_particles

        grad_vB = (-kernel_vB.detach().matmul(v_B_grad) + v_B_gradK) / self.num_particles
        
        grad_clsW = (-kernel_clsW.detach().matmul(clsW_grad) + clsW_gradK) / self.num_particles
        
        grad_clsB = (-kernel_clsB.detach().matmul(clsB_grad) + clsB_gradK) / self.num_particles
        
        # grad_qA, grad_qB, grad_vA, grad_vB = -q_A_grad, -q_B_grad, -v_A_grad, -v_B_grad
        
        return grad_qA, grad_qB, grad_vA, grad_vB, grad_clsW, grad_clsB

    def step_(self):
        
        
        # for n, p in self.net.named_parameters():
        #     if p.requires_grad == True: 
        #         print(n)
        #         p.data = p.data - self.lr * p.grad.data.view(-1).view(p.data.shape)
                
        
        # print('update done')
        # return
        
        
        
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, cls_grad_w, cls_grad_b = self.score_func()
        
        # print(q_A_grad.shape, q_B_grad.shape, v_A_grad.shape, v_B_grad.shape)
        
        
        updated_n = []
        
        for net_id in range(self.num_particles):
            for layer_id in range(12):   
                for n, p in self.net.lora_vit.named_parameters():
                    
                    if p.requires_grad and n not in updated_n: 
                    
                        if f'blocks.{str(layer_id)}' in n:
                            # print('B-name', n)
                            if "proj_q" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    # print(n)
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * q_A_grad[layer_id][net_id].view(p.data.shape)
                                elif f"w_b.layer.{net_id}" in n:
                                    # print(n)
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * q_B_grad[layer_id][net_id].view(p.data.shape)
                            elif "proj_v" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    # print(n)
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * v_A_grad[layer_id][net_id].view(p.data.shape)
                                elif f"w_b.layer.{net_id}" in n:
                                    # print(n)
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * v_B_grad[layer_id][net_id].view(p.data.shape)
                                    
                        elif 'fc' in n:
                            if 'weight' in n:
                                if f"layer.{net_id}" in n:
                                    # print(n)
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * cls_grad_w[layer_id][net_id].view(p.data.shape)
                            elif 'bias' in n and f"layer.{net_id}" in n:
                                    # print(n)
                                    updated_n.append(n)
                                    temp_w = p.data
                                    p.data = temp_w + self.lr * cls_grad_b[layer_id][net_id].view(p.data.shape)
        
        # try:                        
        #     temp_w = self.net.lora_vit.fc.weight.data
        #     temp_b = self.net.lora_vit.fc.bias.data
        # except:
        #     temp_w = self.net.fc.weight.data
        #     temp_b = self.net.fc.bias.data
        
        # try:
        #     self.net.lora_vit.fc.weight.data =  temp_w - self.lr * cls_grad_w
        #     self.net.lora_vit.fc.bias.data =  temp_b - self.lr * cls_grad_b
        # except:
        #     self.net.fc.weight.data =  temp_w - self.lr * cls_grad_w
        #     self.net.fc.bias.data =  temp_b - self.lr * cls_grad_b
            
        # exit()
                                
        
