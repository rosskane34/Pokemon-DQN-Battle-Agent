# Pokémon Double DQN Battle Agent

A **Double Deep Q-Network (DDQN)** agent trained from scratch to master deterministic Pokémon battles in a custom environment.

This project demonstrates end-to-end reinforcement learning: custom Gym-like environment, thoughtful state & reward design, stable DDQN implementation, rigorous evaluation with learning curves + matchup heatmaps, and a playable human-vs-agent mode.

**Key highlight**: Implemented **Double DQN**  to mitigate Q-value overestimation and achieve more reliable convergence.

## Environment

Simplified 1v1 turn-based battles between three Pokémon:

- **Venusaur** (Grass)
- **Charizard** (Fire)
- **Blastoise** (Water)

**Mechanics** (fully deterministic except minor speed-tie randomization):
- 10×10 type chart
- Accurate physical/special damage calculation
- Attack stage modifiers (e.g. Growl)
- Speed-based turn order
- Four moves per Pokémon (damage + status)

No switching, no items, no weather — clean MDP for RL experimentation.

## Double DQN Agent

Core algorithm: **Double Deep Q-Network**  — selects actions using the online/policy network but evaluates them with the target network to reduce overestimation bias.

Implementation details:
- Policy network + separate target network
- Experience replay buffer (capacity 10,000)
- Epsilon-greedy exploration (linear decay)
- Discount factor γ = 0.99
- Huber loss (`SmoothL1Loss`)
- Gradient clipping (max norm 1.0)
- Soft target updates every 300 steps
- Reward normalization → stable Q-values
- No overestimation divergence observed even after long training

**Neural architecture**  
Feed-forward MLP:  
Input (23-dim state) → 64 (ReLU) → 64 (ReLU) → 4 (Q-values per move)

## State Representation (23 dimensions)

Carefully engineered to give the agent full strategic information without being overly large:

**Core battle features (7 dims)**  
- Attacker / defender HP ratios  
- Attacker / defender types (one-hot or normalized)  
- Attacker / defender attack stages  
- Speed priority flag (1 = attacker faster)

**Per-move features (4 moves × 4 = 16 dims)**  
For each move: normalized power, type index, category (physical/special/status), effect flag (stat change)

This compact encoding enables the agent to reason about type matchups, move potency, status utility, and tempo.

## Reward Shaping

Dense, multi-component reward designed for fast & correct learning:

- +damage dealt × 1.5 (scaled)  
- Bonus for super-effective / not-very-effective moves  
- +300 / -300 for KO / being KO'd  
- Small penalty per turn (encourages quick wins)  
- Penalty for using low-damage or zero-damage moves  
- Penalty for misusing status moves 

Rewards clipped & normalized → prevents exploding gradients.

## Training & Results

Trained via self-play over thousands of episodes.

![Training Curve](<img width="995" height="565" alt="Screenshot (419)2" src="https://github.com/user-attachments/assets/759878d6-6803-472f-8b4c-735652484c92" />
)  
*Total reward per episode (blue) + 200-episode moving average (orange). Rapid early improvement, then stable high performance with low variance — no signs of instability.*

![Win-Rate Heatmap](<img width="1600" height="900" alt="Screenshot (420)" src="https://github.com/user-attachments/assets/cd8f260e-aaa9-490d-b822-4a598e059e7f" />
)  
*Empirical win rates (15,000 evaluation episodes) across all 3×3 attacker-defender matchups.*  
Agent perfectly exploits the type triangle (≈1.00 win rate when advantaged, ≈0 when disadvantaged). Mirror matches ≈0.93 due to first-move advantage.

These plots provide clear proof-of-concept: the DDQN policy generalizes correctly and consistently across matchups.

## Play Mode

Run `python main.py --play` after training to battle the agent yourself in a simple text interface.

## Installation & Usage

```bash
# Clone & install
git clone https://github.com/rosskane34/Pokemon-DQN-Battle-Agent.git
cd Pokemon-DQN-Battle-Agent
pip install -r requirements.txt

# Train (takes ~5-10 min on CPU)
python main.py --train

# Play against trained agent
python main.py --play

# Future Directions

- Full Pokémon roster + team building
- Stochastic battles (accuracy, crits)
- PPO, Rainbow, or multi-agent self-play
- Switching mechanics
