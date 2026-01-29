import torch
from env import BlackwallEnv,ACTIONS
from policy import AttentionPolicy

def main():
    env=BlackwallEnv()
    policy=AttentionPolicy(3,3,3)
    policy.load_state_dict(torch.load("blackwall_attn_dqn.pt",weights_only=True))
    state=torch.tensor(env.reset(),dtype=torch.float32).unsqueeze(0)

    while True:
        q=policy(state)[0]
        actions=q.argmax(dim=1).tolist()
        next_state,reward,done=env.step(actions)

        print({
            "actions": [ACTIONS[a] for a in actions],
            "reward": reward
        })

        state=torch.tensor(next_state,dtype=torch.float32).unsqueeze(0)
        if done:
            break

if __name__ == "__main__":
    main()
