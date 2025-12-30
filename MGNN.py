import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from GraphPPIS_model import *


def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Subgraphnet(nn.Module):
    def __init__(self, ks, in_dim, dim, alpha, lamda, act=F.hardtanh, drop_p=0.0):
        super(Subgraphnet, self).__init__()
        self.ks = ks
        self.alpha = alpha
        self.lamda = lamda
        self.fc = nn.Linear(in_dim, dim)
      
        self.act = act
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.subgraph_gcns_1 = nn.ModuleList()
        self.l_n = len(ks)
        self.drop_p = drop_p
        
        for j in range(3):

            self.subgraph_gcns_1.append(GraphConvolution(dim, dim, variant=True,residual=True))# ,dropout=self.drop_p)) # 16 20
  
        for i in range(self.l_n):
           
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def readout(self, hs):
        h_max = torch.Tensor([torch.max(h, 0)[0] for h in hs])

        return h_max

    def forward(self, feat, edge , ep):
    
        hs = []
        org_h = nn.ReLU()(self.fc(feat))
        org_g = edge
        
        for i in range(self.l_n):
            g = org_g
            h = org_h
            # print(h.shape, g.shape)

            if self.ks[i] != 1:
                g, h, idx = self.pools[i](g, h, ep)
                # print(h.shape, g.shape)
                # exit()
         
            h0 = h
            for j in range(3):
                h_temp =  self.subgraph_gcns_1[j](h, g, h0, self.lamda, self.alpha, j+1)
                h = torch.relu(h + h_temp)
                
            if self.ks[i] != 1:
                g, h = self.unpools[i](org_g, h, org_h, idx)

            hs.append(h)
        
        for h_i in hs:
            h = torch.max(h,h_i)
 
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h, ep):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        
        return top_k_graph(scores, g, h, self.k,ep)

class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k, ep):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)#

    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()#
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = un_g
    return g, new_h, idx




