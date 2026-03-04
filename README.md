# Blackwall - Self-Healing ML Control System

Blackwall is a **self-healing machine learning control system** that learns *when* to intervene in production ML pipelines instead of blindly retraining models.

It separates **prediction** from **adaptation** by introducing a learned control plane that governs multiple independent ML models under drift, cost, and uncertainty. We use Reinforcement Learning (DQN) to determine the best actions to be taken based on collected metrics.


### Supported actions: -
- **OBSERVE** - no action taken  
- **FINETUNE** - lightweight retraining  
- **RETRAIN** - full retraining  

## Architecutre Diagram: -
<img width="400" height="7596" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/c80abba0-3be0-4bf1-9fe1-43fe57213c13" />


The system optimizes the tradeoff between **model confidence** and **retraining cost**.

## Execution Flow: -

1. Child models process incoming data streams.
2. Health signals are computed per model.
3. Attention-based policy evaluates all models jointly.
4. Policy selects actions for each model.
5. Retraining is triggered only when beneficial.
6. Training jobs are executed asynchronously through the backend service.
7. Updated models are loaded back into the system.
8. System continues without global restarts.

## Observed Behavior: -

During training and inference:

- The policy **prefers OBSERVE** when confidence is high.
- Retraining is applied **selectively**, not globally.
- Different child models receive different actions.
- Increasing drift severity or lowering retraining cost leads to more intervention.

This behavior is **intentional** and reflects real-world cost-aware control systems.

---

### AWS Mapping (Conceptual)

| Blackwall Component | AWS Service |
|--------------------|-------------|
| Control Plane | ECS / SageMaker Endpoint |
| XGBoost Training | SageMaker Training Jobs |
| Model Artifacts | S3 |
| Retraining Triggers | EventBridge |
| Logging | CloudWatch |
| Infrastructure | Terraform |

Infrastructure is **not applied** due to cost constraints.  
Terraform files are included to demonstrate **infrastructure design and lifecycle alignment**, not live deployment.

---

## Infrastructure as Code

Terraform definitions are provided under `infra/aws/` to model:
- Control-plane runtime
- Training job orchestration
- Artifact storage

These definitions are **conceptual** and intended for design clarity.




## Key Takeaway

Blackwall demonstrates that **inaction can be optimal**.

By learning *when not to retrain*, the system reduces cost,instability, and unnecessary churn. 

---

Tested main.py for synthetic data should look like 
<img width="991" height="246" alt="image" src="https://github.com/user-attachments/assets/8d6454db-1901-4974-bba5-3fcc6e4021de" />
## How to Run Locally

```bash
python test.py            # sanity check
python train_policy.py    # train control policy
python main.py            # run trained system






