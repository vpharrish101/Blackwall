import numpy as np
import pandas as pd

def generate(n=1200,
             drift=False,
             seed=73):
    
    rng=np.random.default_rng(seed)

    latency=rng.normal(100,10,n)
    error_rate=rng.beta(1,25,n)
    volume=rng.normal(10_000,800,n)

    if drift:
        latency+=np.linspace(0,80,n)
        error_rate+=np.linspace(0,0.25,n)

    failure=(latency>180)|(error_rate>0.3)
    return pd.DataFrame({
        "latency":latency,
        "error_rate":error_rate,
        "volume":volume,
        "failure":failure.astype(int)
        })

if __name__=="__main__":
    train=generate(drift=False)
    stream=generate(drift=True)
    train.to_csv(r"D:\Python311\Pets\Blackwall\data\train.csv",index=False)
    stream.to_csv(r"D:\Python311\Pets\Blackwall\data\stream.csv",index=False)
