import numpy as np
from data_simul import generate
from child import train_child,predict_child
from metrics import psi,entropy

ACTIONS=["OBSERVE","FINETUNE","RETRAIN"]
COST=[0.0,0.01,0.02]

class BlackwallEnv:
    def __init__(self,window=120,n_children=3):
        self.window=window
        self.n=n_children
        self.reset()

    def reset(self):
        self.ptr=0
        self.drifts=np.random.uniform(0.7,1.2,self.n)

        self.streams=[generate(drift=d,seed=i) for i,d in enumerate(self.drifts)]

        ref=generate(drift=0.0,seed=999) #type:ignore
        self.X_ref=ref[["latency","error_rate","volume"]]
        self.y_ref=ref["failure"]

        self.models=[train_child(self.X_ref,self.y_ref) for _ in range(self.n)]

        return self._state()

    def _child_state(self,i):
        df=self.streams[i]
        win=df.iloc[self.ptr:self.ptr+self.window]
        X=win[["latency","error_rate","volume"]]

        probs=predict_child(self.models[i],X)

        return np.array([
            psi(self.X_ref["latency"],X["latency"]),
            entropy(probs),
            self.drifts[i]
        ])

    def _state(self):
        return np.stack([self._child_state(i) for i in range(self.n)])

    def step(self,actions):
        cost=sum(COST[a] for a in actions)

        for i,a in enumerate(actions):
            if a>0:
                self.models[i]=train_child(self.X_ref,self.y_ref)

        self.ptr+=self.window
        done=self.ptr+self.window>=len(self.streams[0])

        next_state=self._state()
        reward=-next_state[:,1].mean()-cost

        return next_state,reward,done
