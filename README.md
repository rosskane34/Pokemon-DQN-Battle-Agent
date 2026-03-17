# Pokemon-DQN-Battle-Agent
A custom Deep Q‑Network (DQN) agent trained to battle in a simplified Pokémon environment.
This project includes:

- A fully custom turn‑based battle environment

- A PyTorch DQN implementation with replay buffer + target network

- A neural‑encoded state representation (HP, type, stages, move features)

- A training loop with reward shaping

- A playable mode where the trained agent battles the user

# Environment Overview
The environment simulates deterministic Pokémon battles between:

- Venusaur (Grass)

- Charizard (Fire)

- Blastoise (Water)

Each Pokémon includes:

- Base stats (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed)

- Four moves (physical, special, or status)

- A type from a 10‑type system

- Attack stage modifiers (e.g., Growl reduces Attack)

# Battle Mechanics
- 10×10 type‑effectiveness matrix -> # More Types to be added later 

- Physical and special damage formulas

- Status effects (currently: Attack‑lowering)

- Turn order determined by Speed

- Deterministic transitions (no RNG)

The agent selects one of four moves each turn with the goal of defeating the opponent efficiently.

# Deep Q‑Network Agent
The agent uses a standard but fully engineered DQN setup:

- Policy network + target network

- Replay buffer (10,000 transitions)

- Epsilon‑greedy exploration

- Discount factor γ = 0.90

- Huber loss for stability

- Gradient clipping

- Target network updates every 300 steps

- Reward normalization to prevent Q‑value explosion

# Neural Architecture
A feed‑forward network:
- Input:  23‑dimensional state vector
- Hidden: 64 → 64 (ReLU)
- Output: 4 Q‑values (one per move)

# State Representation (23 Dimensions)
The environment encodes each battle state as a 23‑dimensional vector.
# 1. Core Battle Features (7)
- [0] Attacker HP ratio
- [1] Defender HP ratio
- [2] Attacker type (normalized index)
- [3] Defender type (normalized index)
- [4] Attacker attack stage (normalized)
- [5] Defender attack stage (normalized)
- [6] Speed flag (1 if attacker moves first)
# 2. Move Features (16)
- Power (normalized)
- Type (normalized)
- Category (0=physical, 0.5=special, 1=status)
- Effect flag (1 if stat‑changing)

# Total: 7 + 16 = 23 features
This representation gives the agent enough information to reason about:

- Type matchups

- Move strength

- Status utility

- Turn order

- Battle progression

# Reward Shaping
The reward function encourages:

- Dealing damage

- Using super‑effective moves

- Finishing battles quickly

- Avoiding weak or zero‑damage moves

- Using status moves strategically

- Rewards are normalized to stabilize Q‑values.

# Training
Training is performed over thousands of self‑play episodes.
Each episode ends when one Pokémon faints.

A trained model is saved as:
- dqn_pokemon.pth

# Play Mode
After training, the agent can battle the user in a turn‑based interface.

# Future Work
- Add more Pokémon and move diversity

- Expand the type system

- Add stochasticity (crit chance, accuracy)

- Implement multi‑agent self‑play

- Explore PPO or Rainbow DQN variants
