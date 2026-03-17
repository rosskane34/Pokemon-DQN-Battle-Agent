# Pokemon-DQN-Battle-Agent
A custom Deep Q‑Network (DQN) agent trained to battle in a simplified Pokémon environment.
This project includes:

- A fully custom turn‑based battle environment

- A PyTorch DQN implementation with replay buffer + target network

- A neural‑encoded state representation (HP, type, stages, move features)

- A training loop with reward shaping

- A playable mode where the trained agent battles the user

# Project Description
- The environment simulates simplified Pokémon battles between three starter Pokémon: Venusaur, Charizard, and Blastoise. Each Pokémon has:

- Base stats (HP, Attack, Defense, Special Attack, Special Defense, Speed)

- Four moves (physical, special, or status)

- A type (Grass, Fire, Water, etc.)

- Attack stage modifiers (e.g., Growl reduces Attack)

# The battle system includes:

- A 10×10 type‑effectiveness matrix

- Physical and special damage formulas

- Status effects (currently: Attack‑lowering)

- Turn order determined by Speed

- Deterministic damage (no randomness)

The agent selects one of four moves each turn. The objective is to defeat the opponent as efficiently as possible while learning effective move selection strategies.

# Deep Q-Network Agent
The agent uses a standard DQN setup:

- Policy network and target network

- Replay buffer for experience storage

- Epsilon‑greedy exploration

- Huber loss for stability

- Gradient clipping

- Periodic target network updates

- Discount factor γ = 0.99

The neural network is a feed‑forward model with two hidden layers (ReLU activations) mapping a 23‑dimensional state vector to four Q‑values (one per move).

# State Representation
The environment encodes each battle state as a 23‑dimensional vector. This representation includes both Pokémon status information and move‑level features.

# Core Battle Features (7 values)
- Player HP ratio (0–1)

- Enemy HP ratio (0–1)

- Player type (normalized index)

- Enemy type (normalized index)

- Player attack stage (normalized)

- Enemy attack stage (normalized)

- Speed flag (1 if player is faster, else 0)

These features summarize the essential battle conditions.

# Move Features (16 values)
Each of the player's four moves contributes four features:

- Power (normalized by dividing by 100)

- Type (normalized index)

- Category (0 = physical, 1 = special, 2 = status; normalized)

- Effect flag (1 if the move applies a stat change, else 0)

This gives the agent enough information to reason about move strength, typing, category, and utility.

# Total State Size
7 core features+ 4 moves × 4 features each = 23-dimensional state vector


# Training
Training is performed over many episodes of self‑play. Each episode continues until one Pokémon faints. The agent receives shaped rewards to encourage:

- Dealing damage

- Using super‑effective moves

- Finishing battles quickly

- Avoiding ineffective or zero‑damage moves

- Using status moves effectively

A trained model is saved as:
- # dqn_pokemon.pth

# Play Mode
After training, the agent can be evaluated in a human‑playable mode. The user selects a Pokémon and chooses moves manually, while the agent selects moves greedily based on learned Q‑values. Debug functions are included to print state vectors and Q‑values for analysis.
