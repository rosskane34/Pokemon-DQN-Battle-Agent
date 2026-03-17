"""
Battle Environment for Pokemon RL AI
"""
from dqn_agent import DQNAgent
import random
import numpy as np
import torch
from PokemonClass import Pokemon
from Moves import Move
import matplotlib.pyplot as plt
import seaborn as sns




# ---------------- TYPE SYSTEM ---------------- #
"""
Typing matrix implemented to handle various types of damage.
Future works will add more pokemon of different types

first index will represent the attacking moves type
second index represents the defenders pokemon type 
"""
type_matrix = np.array([
    [0.5,2.0,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0],   # water    
    [0.5,0.5,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],   # fire
    [2.0,0.5,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0],   # grass
    [1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0],   # normal
    [2.0,1.0,0.5,1.0,0.5,1.0,1.0,1.0,1.0,1.0],   # electric
    [1.0,1.0,1.0,1.0,1.0,0.5,2.0,1.0,1.0,2.0],   # dark
    [1.0,1.0,1.0,0.0,1.0,0.5,2.0,1.0,1.0,2.0],   # ghost
    [1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0],   # dragon
    [1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,0.5,2.0],   # fighting
    [1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,2.0,0.5]    # psychic
])

type_indices = {
    "Water":0,"Fire":1,"Grass":2,"Normal":3,"Electric":4,
    "Dark":5,"Ghost":6,"Dragon":7,"Fighting":8,"Psychic":9
}


# ---------------- POKEMON FACTORY ---------------- #

def getPokemon(num:int):

    if num == 1:
        return Pokemon("Venusaur","Grass",80,82,83,100,100,80,
                       ["Growl","Tackle","Vine Whip","Bite"])

    if num == 2:
        return Pokemon("Charizard","Fire",78,84,78,109,85,100,
                       ["Growl","Tackle","Flamethrower","Bite"])

    if num == 3:
        return Pokemon("Blastoise","Water",79,83,100,85,105,78,
                       ["Growl","Tackle","Waterfall","Bite"])


# ---------------- DAMAGE SYSTEM ---------------- #

def get_attack_stage_multiplier(stage):

    if stage >= 0:
        return (2 + stage) / 2
    else:
        return 2 / (2 - stage)


def get_type_multiplier(move_type, defender_type):

    atk = type_indices[move_type]           #determine the type of move the attack is
    defend = type_indices[defender_type]    #and the type of pokemon the defender is

    return type_matrix[atk][defend]         #return our damage multiplier


def determineEffectivenessType(multiplier):

    if multiplier > 1:                      #print to user if attack was super effective or not
        print("Super Effective!\n")

    if multiplier < 1:
        print("Not very effective...\n")


# ---------------- STATE REPRESENTATION ---------------- #

'''
The following functions are used to determine player and enemy health
as well as the type of both player and enemy
'''

def hp_bucket(currentHp,maxHP):

    ratio = currentHp / maxHP               #we use a system of 2 for High health, 1 for mid health and 0 for low health
    if(ratio>=0.7):
        return 2        
    elif(ratio <0.7 and ratio>=0.4):
        return 1
    else:
        return 0
    
def encode_move(move):
    return [
        move.getPower() / 100.0,                  # normalize power
        type_indices[move.getType()] / 10.0,      # normalize type
        move.getCategory() / 2.0,                 # 0,1,2 → 0–1
        move.getEffect()                          
    ]

def getBattleState(player, enemy):


    player_hp = player.get_currentHp() / player.get_MaxHP() #store player and enemy hp
    enemy_hp  = enemy.get_currentHp() / enemy.get_MaxHP()

    player_type = type_indices[player.get_type()] / 10.0    #store player and enemy type
    enemy_type  = type_indices[enemy.get_type()] / 10.0

    player_attack_stage = player.get_Attack_Stage() / 6.0   #store attack stage for player and enemy
    enemy_attack_stage  = enemy.get_Attack_Stage() / 6.0

    speed_flag = 1.0 if player.get_Speed() >= enemy.get_Speed() else 0.0    #store a flag if player is faster than enemy

    state = [
        player_hp,
        enemy_hp,
        player_type,
        enemy_type,
        player_attack_stage,
        enemy_attack_stage,
        speed_flag
    ]

    # --- Move features (4 moves) ---
    for move in player.get_moves():
        state.extend(encode_move(move))
    return np.array(state, dtype=np.float32)

def compute_reward(move, attacker, defender, old_hp, new_hp, multiplier, fainted):
    reward = 0

    damage = old_hp - new_hp
    max_hp = defender.get_MaxHP()

    # --- Damage reward ---
    reward += damage * 1.5

    # --- Move power reward ---
    reward += move.getPower() * 0.02

    # --- Type effectiveness reward ---
    if multiplier > 1.0:
        reward += 150 * multiplier
    elif multiplier == 1.0:
        reward += 0
    elif multiplier > 0.0:
        reward -= 40
    else:
        reward -= 120

    # --- Status effect reward ---
    if move.getEffect() == 1:
        before = defender.get_Attack_Stage()
        after = defender.get_Attack_Stage()
        if after < before:
            reward += 1

    # --- KO reward ---
    if fainted:
        reward += 300

    # --- Long battle penalty ---
    reward -= 8

    # --- NEW: Penalize weak moves ---
    if damage < 0.10 * max_hp:   # less than 10% HP damage
        reward -= 25

    # --- NEW: Penalize using status moves when KO is possible ---
    if move.getCategory() == 2 and old_hp <= attacker.get_Attack() * 2:
        reward -= 40

    return reward/50
# ---------------- Visualization ---------------- #
def moving_avg(data, window=200):
    return np.convolve(data, np.ones(window)/window, mode='valid')

pokemon_list = ["Charizard", "Venusaur", "Blastoise"]
index = {name: i for i, name in enumerate(pokemon_list)}

wins = np.zeros((3, 3))
battles = np.zeros((3, 3))





# ---------------- TRAINING LOOP ---------------- #

def run_training():

    # Create the DQN agent (23 inputs = state size, 4 outputs = actions)
    agent = DQNAgent(state_size=23, action_size=4)
    episode_rewards=[]
    # agent.load()  # Uncomment if you want to resume from a saved model
    
    EPISODES = 15000  #How many times we wish to train

    for episode in range(EPISODES):
        total_reward=0
        p1 = getPokemon(random.randint(1,3))        #randomly choose two pokemon
        p2 = getPokemon(random.randint(1,3))

        while p1.get_currentHp() > 0 and p2.get_currentHp() > 0:

            # Determine turn order
            if p1.get_Speed() >= p2.get_Speed():
                first, second = p1, p2
            else:
                first, second = p2, p1

            # -------- FIRST ATTACK --------
            reward = 0

            state = getBattleState(first, second)
            action = agent.choose_action(state)     # DQN chooses action
            move = first.choose_Move(action+1)

            old_hp = second.get_currentHp()     #save old Hp to do damage calculations

       

            # Determine correct offensive and defensive stats
            if move.getCategory() == 0:  # Physical
                atk_stat = first.get_Attack()
                def_stat = second.get_Defense()
            elif move.getCategory() == 1:  # Special
                atk_stat = first.get_specialAttack()
                def_stat = second.get_specialDefense()
            else:  # Status moves do no damage
                atk_stat = 0
                def_stat = 1

            # Apply attack stage multiplier
          
            atk_stat *= get_attack_stage_multiplier(first.get_Attack_Stage())

            # Compute damage
            multiplier = get_type_multiplier(move.getType(), second.get_type())
            damage = (atk_stat / def_stat) * move.getPower()
            damage *= multiplier
            second.take_Damage(damage)

            fainted = second.get_currentHp() <= 0
            reward = compute_reward(
                move=move,
                attacker=first,
                defender=second,
                old_hp=old_hp,
                new_hp=second.get_currentHp(),
                multiplier=multiplier,
                fainted=fainted
            )

            next_state = getBattleState(first, second)
            total_reward+=reward 
            # ─── DQN update ───
            done = second.get_currentHp() <= 0
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()   # This performs batch training when buffer is ready

            if fainted==True:
                break


            # -------- SECOND ATTACK --------
            reward = 0

            state = getBattleState(second, first)
            action = agent.choose_action(state)
            move = second.choose_Move(action+1)

            old_hp = first.get_currentHp()

            # Status effects
            effect = move.getEffect()

            # Determine correct offensive and defensive stats
            if move.getCategory() == 0:  # Physical
                atk_stat = second.get_Attack()
                def_stat = first.get_Defense()
            elif move.getCategory() == 1:  # Special
                atk_stat = second.get_specialAttack()
                def_stat = first.get_specialDefense()
            else:  # Status moves do no damage
                atk_stat = 0
                def_stat = 1

            # Apply attack stage multiplier only to physical moves
            if move.getCategory() == 0:
                atk_stat *= get_attack_stage_multiplier(second.get_Attack_Stage())

            # Compute damage
            multiplier = get_type_multiplier(move.getType(), first.get_type())
            damage = (atk_stat / def_stat) * move.getPower()
            damage *= multiplier
            first.take_Damage(damage)
            #Call our reward Calculator Function
            fainted = first.get_currentHp() <= 0
            reward = compute_reward(
            move=move,
            attacker=second,
            defender=first,
            old_hp=old_hp,
            new_hp=first.get_currentHp(),
            multiplier=multiplier,
            fainted=fainted
            )
            total_reward+=reward  #keep track of this episodes total reward

            next_state = getBattleState(second, first)

            # DQN update, if the enemy is dead pass that information on to the agent as well
            done = first.get_currentHp() <= 0
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            if first.get_currentHp() <= 0:
                break
        episode_rewards.append(total_reward)        # get total episode reward

        att = index[p1.get_name()]
        defe = index[p2.get_name()]

        battles[att, defe] += 1

        if p1.get_currentHp() > 0:
            wins[att, defe] += 1


        # Epsilon decay printing
        if episode % 100 == 0:
            print("Episode:", episode, "Epsilon:", agent.epsilon)

    agent.save()
    win_rates = np.divide(wins, battles, out=np.zeros_like(wins), where=battles > 0)    #compute the win rates for each pokemon
    print("Training complete.")
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, alpha=0.3, label="Episode Return")
    plt.plot(moving_avg(episode_rewards, 200), color='red', label="200-episode Moving Average") # graph for 200 episode moving average of the reward for our moves
    plt.title("Training Performance Over 15,000 Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.show()

   
    plt.figure(figsize=(6,5))
    sns.heatmap(win_rates, annot=True, cmap="Blues", xticklabels=pokemon_list, yticklabels=pokemon_list)
    plt.title("Win Rate Heatmap (Attacker vs Defender)")                    
    plt.xlabel("Defender")                                     #heatmap of victories for when pokemon is player1
    plt.ylabel("Attacker")
    plt.show()



# ---------------- GAMEPLAY ---------------- #
def play_game():

    agent = DQNAgent(state_size=23, action_size=4)      #load our agent
    agent.load()
    agent.epsilon = 0.0

    print("\n=== Pokémon Battle RL ===\n")
    print("Choose your Pokémon:")           #prompt user to pick a pokemon
    print("1. Venusaur")
    print("2. Charizard")
    print("3. Blastoise")

    player_choice = int(input("\nEnter choice (1-3): "))
    player = getPokemon(player_choice)

    enemy = getPokemon(random.randint(2, 2))        #randomly generate enemy pokemon

    print("\n=== Enemy Pokémon ===")
    enemy.display_Pokemon()

    print("\n=== Your Pokémon ===")
    player.display_Pokemon()            #display enemy pokemon and your pokemon

    firstString=""
    secondString=""
    # Determine turn order
    if player.get_Speed() > enemy.get_Speed():
        first = player
        second = enemy
        firstString="Player"
        secondString="Computer"
    elif player.get_Speed() < enemy.get_Speed():
        first = enemy
        second = player
        firstString="Computer"
        secondString="Player"
    else:                                   #randomly choose who goes first if speeds are the same
        randomChoice=random.randint(1,2)
        if(randomChoice==1):
            first = player
            second = enemy
            firstString="Player"
            secondString="Computer"
        else:
            first = enemy
            second = player
            firstString="Computer"
            secondString="Player"

    print("\nBattle Start!\n")
    
    while player.get_currentHp() > 0 and enemy.get_currentHp() > 0:

        # ---------------- FIRST ATTACK ----------------
        print("--------------------------------------------------")
        print(f"{firstString} moves first.")

        if first == player:
            print("\nYour Move Options:")
            player.show_Moves()                     #prompt user for their moves
            choice = int(input("Select move (1-4): "))
            move = player.choose_Move(choice)
            print(f"\nYou used {move.getName()}.")
        else:
            state = getBattleState(enemy, player)
            ai_action = agent.choose_action(state, force_greedy=True)   #enemy chooses their moves
            move = enemy.choose_Move(ai_action + 1)
            print(f"\nEnemy used {move.getName()}.")

      
        # Damage calculation
        if move.getCategory() == 0:
            atk_stat = first.get_Attack() * get_attack_stage_multiplier(first.get_Attack_Stage()) #get attackers attack Stat
            def_stat = second.get_Defense()
        elif move.getCategory() == 1:           #get correct defense/attack stat depending on move type
            atk_stat = first.get_specialAttack()
            def_stat = second.get_specialDefense()
        else:
            atk_stat = 0
            def_stat = 1

        multiplier = get_type_multiplier(move.getType(), second.get_type())
        damage = (atk_stat / def_stat) * move.getPower() * multiplier

        determineEffectivenessType(multiplier)       #determine if move was super effective or not
        second.take_Damage(damage)

        print("\nCurrent Status:")
        print(firstString)
        print("==========")
        first.show_Battle_Status()
        print(f"\n{secondString}")
        print("==========")
        second.show_Battle_Status()

        if second.get_currentHp() <= 0:
            print(f"\n{second.get_name()} fainted.")
            break

        # ---------------- SECOND ATTACK ----------------
        print("--------------------------------------------------")
        print(f"{secondString} moves next.")

        if second == player:
            print("\nYour Move Options:")
            player.show_Moves()                     #prompt user for input
            choice = int(input("Select move (1-4): "))
            move = player.choose_Move(choice)
            print(f"\nYou used {move.getName()}.")
        else:
            state = getBattleState(enemy, player)
            ai_action = agent.choose_action(state, force_greedy=True)
            move = enemy.choose_Move(ai_action + 1)
            print(f"\nEnemy used {move.getName()}.")
       
        # Damage calculation
        if move.getCategory() == 0:
            atk_stat = second.get_Attack() * get_attack_stage_multiplier(second.get_Attack_Stage())
            def_stat = first.get_Defense()
        elif move.getCategory() == 1:
            atk_stat = second.get_specialAttack()
            def_stat = first.get_specialDefense()
        else:
            atk_stat = 0
            def_stat = 1

        multiplier = get_type_multiplier(move.getType(), first.get_type())
        damage = (atk_stat / def_stat) * move.getPower() * multiplier

        determineEffectivenessType(multiplier)
        first.take_Damage(damage)

        print("\nCurrent Status:")
        print(firstString)
        print("==========")
        first.show_Battle_Status()
        print(f"\n{secondString}")
        print("==========")
        second.show_Battle_Status()

        if first.get_currentHp() <= 0:
            print(f"\n{first.get_name()} fainted.")
            break

    print("\n=== Battle Over ===\n")


def debug_print_state(state):

    print("\n=== STATE DEBUG ===")
    print(f"Player HP: {state[0]:.3f}")
    print(f"Enemy HP: {state[1]:.3f}")
    print(f"Player Type: {state[2]:.3f}")
    print(f"Enemy Type: {state[3]:.3f}")
    print(f"Player Atk Stage: {state[4]:.3f}")
    print(f"Enemy Atk Stage: {state[5]:.3f}")
    print(f"Speed Flag: {state[6]}")

    print("\n--- Moves ---")
    idx = 7
    for i in range(4):
        power = state[idx]
        mtype = state[idx+1]
        cat = state[idx+2]
        effect = state[idx+3]

        print(f"Move {i+1}: Power={power:.2f}, Type={mtype:.2f}, Cat={cat:.2f}, Effect={effect}")
        idx += 4

    print("====================\n")

def debug_q_values(agent, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            qvals = agent.policy_net(state_tensor)[0].numpy()

        print("\n=== Q VALUES ===")
        for i, q in enumerate(qvals):
            print(f"Move {i+1}: {q:.3f}")
        print("================\n")
