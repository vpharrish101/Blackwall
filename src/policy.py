import torch
import torch.nn as nn

class AttentionPolicy(nn.Module):
    def __init__(self,state_dim,n_children,n_actions):
        super().__init__()
        self.embed=nn.Linear(state_dim,32)
        self.attn=nn.MultiheadAttention(32,num_heads=4,batch_first=True)
        self.heads=nn.ModuleList([nn.Linear(32,n_actions) for _ in range(n_children)])

    def forward(self,x):
        z=self.embed(x)
        z,_=self.attn(z,z,z)
        qs=[]
        for i,head in enumerate(self.heads):
            qs.append(head(z[:,i]))
        return torch.stack(qs,dim=1)
