# Improving Robustness by Action Correction via Multi-step Maximum Risk Estimation

This repository contains the code for the paper **"Improving Robustness by Action Correction via Multi-step Maximum Risk Estimation"**. It provides implementations for training and testing agents using Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO) algorithms with options for vanilla and robust training strategies.

## Requirements

```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone https://github.com/tony-soochow/MMRAR.git
cd MMRAR
```

## Training

To train the agent using PPO or TRPO:

1. Select the Training Strategy:
   - Open `train_ppo.py` or `train_trpo.py`.
   - Modify the import statement to choose the desired agent:
     ```python
     # For vanilla PPO
     from agent.ppo import PPO
     # For robust PPO
     from agent.robust_adv_ppo_myworsttwostepsR import PPO
     ```

2. Run the Training Script:
   ```bash
   python train_ppo.py
   python train_trpo.py
   ```

## Testing

To test the agent's performance, use `all_ppo_inference.py` or `all_trpo_inference.py`:

1. Select the Testing Mode:
   - Open `all_ppo_inference.py` or `all_trpo_inference.py`.
   - Set the testing mode in the code:
     ```python
     # For no attack
     choices = 'Nominal'
     # For attack
     choices = 'attack'
     ```

2. Run the Testing Script:
   ```bash
   python all_ppo_inference.py
   python all_trpo_inference.py
   ```
