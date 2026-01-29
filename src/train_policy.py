import torch
from torch import optim
import numpy as np
from collections import deque
import random

from env import BlackwallEnv
from policy import AttentionPolicy

env=BlackwallEnv()
policy=AttentionPolicy(state_dim=3,n_children=3,n_actions=3)
target=AttentionPolicy(3,3,3)
target.load_state_dict(policy.state_dict())

opt=optim.Adam(policy.parameters(),lr=1e-3)#type:ignore
buffer=deque(maxlen=10_000)

GAMMA=0.99
EPS=1.0

def sample_batch(batch_size):
    batch=random.sample(buffer,batch_size)

    states=torch.tensor(np.stack([b[0] for b in batch]),dtype=torch.float32)  
    actions=torch.tensor(np.stack([b[1] for b in batch]),dtype=torch.long)  

    rewards=torch.tensor([b[2] for b in batch],dtype=torch.float32)  

    next_states=torch.tensor(np.stack([b[3] for b in batch]),dtype=torch.float32)
    dones=torch.tensor([b[4] for b in batch],dtype=torch.bool) 

    return states,actions,rewards,next_states,dones

for ep in range(320):
    s=torch.tensor(env.reset(),dtype=torch.float32).unsqueeze(0)
    total=0

    while True:
        if np.random.rand()<EPS:
            a=torch.randint(0,3,(3,))
        else:
            q=policy(s)[0]
            a=q.argmax(dim=1)

        ns,r,done=env.step(a.tolist())
        buffer.append((
            s.squeeze(0).numpy(),  
            a.numpy(),             
            r,                      
            ns,                     
            done                    
        ))

        s=torch.tensor(ns,dtype=torch.float32).unsqueeze(0)
        total+=r

        if len(buffer)>64:
            bs,ba,br,bns,bd=sample_batch(64)
            bs=bs.float()
            bns=bns.float()
            q=policy(bs)
            q=q.gather(2,ba.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                tq=target(bns).max(dim=2)[0]

            loss=((br.unsqueeze(1)+GAMMA*tq*(~bd.unsqueeze(1)))-q).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if done:
            break

    EPS=max(0.05,EPS*0.995)
    if ep%10==0:
        target.load_state_dict(policy.state_dict())

    #print(f"Episode {ep} | Reward {total:.2f}")

torch.save(policy.state_dict(),"blackwall_attn_dqn.pt")
