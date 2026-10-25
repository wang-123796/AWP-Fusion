import torch
import torch.nn as nn
import torch.nn.functional as F

class RDBlock(nn.Module):
    def __init__(self, dim):
        super(RDBlock, self).__init__()
        self.dense = nn.Linear(dim, dim)
        
    def forward(self, x):
        x0 = x
        x = F.leaky_relu(self.dense(x))       
        x = x0 + x
        return x

class MultiAttModel(nn.Module):
    def __init__(self, dim, window, n_head, n_RD):
        super(MultiAttModel, self).__init__()
        self.n_RD = n_RD
        self.n_head = n_head
        
        # Projection layer: 1280 -> 320
        self.proj = nn.Linear(1280, 320)
        
        self.cnn_v = nn.Conv1d(320, 320, kernel_size=2*window+1, padding=window)
        self.W_cnns = nn.ModuleList([nn.Conv1d(320, 320, kernel_size=2*window+1, padding=window) for _ in range(n_head)])
        self.RDs = nn.ModuleList([RDBlock(2*n_head*320) for _ in range(n_RD)])  
        self.output = nn.Linear(2*n_head*320, 1)
        
    def forward(self, emb, mask=None):
        emb = emb.permute(0, 2, 1)  # (B, L, 1280)
        emb = self.proj(emb)        # (B, L, 320)
        emb = emb.permute(0, 2, 1)  # (B, 320, L)
        
        values = self.cnn_v(emb)
        for i in range(self.n_head):
            weights = self.W_cnns[i](emb)
            
            # Mask padding tokens
            if mask is not None:
                weights = weights.masked_fill(~mask.unsqueeze(1), -1e9)
            
            weights = F.softmax(weights, dim=-1)
            x_sum = torch.sum(values * weights, dim=-1)      # Sum pooling
            x_max, _ = torch.max(values * weights, dim=-1)   # Max pooling
            
            if i == 0:
                cat_xsum = x_sum
                cat_xmax = x_max
            else:
                cat_xsum = torch.cat([cat_xsum, x_sum], dim=1)
                cat_xmax = torch.cat([cat_xmax, x_max], dim=1)
                
        cat_f = torch.cat([cat_xsum, cat_xmax], dim=1)       # Concat features for regression
        for j in range(self.n_RD):
            cat_f = self.RDs[j](cat_f)
            
        return self.output(cat_f)