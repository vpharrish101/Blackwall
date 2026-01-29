import numpy as np

def psi(ref,cur,bins=10):
    rh,edges=np.histogram(ref,bins=bins)
    ch,_=np.histogram(cur,bins=edges)

    rp=rh/(rh.sum()+1e-6)
    cp=ch/(ch.sum()+1e-6)

    return np.sum((cp-rp)*np.log((cp+1e-6)/(rp+1e-6)))

def entropy(p):
    eps=1e-9
    return -np.mean(p*np.log(p+eps)+(1-p)*np.log(1-p+eps))
