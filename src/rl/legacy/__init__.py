"""
Legacy code that references outdated APIs.

This folder contains code that was part of an earlier iteration of the RL system.
The code here is kept for reference but is NOT actively maintained.

Contents:
- dqn.py: Deep Q-Network model (uses old encoding API)
- evaluation.py: Evaluation functions (uses old game combat API)
- test.py: Test utilities (uses old game combat API)
- dqn_algorithm/: DQN training code (uses old APIs)

To use this code, you would need to:
1. Update the imports to match the current API
2. Fix any broken references to moved/deleted modules
3. Update encoding functions to match current state representation

For the current working implementation, see:
- src/rl/models/actor_critic.py (main model)
- src/rl/algorithms/actor_critic/ (training)
"""
