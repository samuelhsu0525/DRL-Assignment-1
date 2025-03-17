import numpy as np
import pickle
import random
import gym

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_action_probs(action_count, exponent=2):
    rank = np.max(action_count) - action_count + 1
    action_probs = softmax(rank ** exponent)
    if np.sum(action_probs) == 0:
        return np.ones_like(action_probs) / len(action_probs)
    return action_probs

def no_prev_action(obs, prev_dir):
    if prev_dir is None:
        action = random.randint(0, 3)
    else:
        action = random.choice([a for a in [0, 1, 2, 3] if a != prev_dir])
    return prev_dir, action

def encode_taxi_v3_state(obs):
    taxi_row, taxi_col, _, _, _, _, _, _, _, _, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    # Map passenger_look to index (0-4)
    location_map = {(0,0): 0, (0,4): 1, (4,0): 2, (4,3): 3}
    passenger_idx = location_map.get(passenger_look, 4)  # If not found, assume inside taxi (4)

    # Map destination_look to index (0-3)
    destination_idx = location_map.get(destination_look, 0)  # Default to 0 if not found

    # Compute the Taxi-v3 state
    state = ((taxi_row * 5 + taxi_col) * 5 + passenger_idx) * 4 + destination_idx
    
    return state

def train(alpha=0.1, gamma=0.6, epsilon_start=1, epsilon_end = 0.1, episodes=1000, decay_rate=0.999):
    global q_table
    print("Training Q-table...")
    env = gym.make("Taxi-v3")
    epsilon = epsilon_start

    q_table = {}  # Initialize Q-table

    for episode in range(episodes):
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})  # Handle Gym v0.26+
        
        total_reward = 0
        done = False
        while not done:
            if obs not in q_table:
                q_table[obs] = np.zeros(6)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[obs])

            next_obs, reward, done, truncated, _ = env.step(action)  # Correct unpacking of `step()`
            total_reward += reward

            if truncated:
                break

            if next_obs not in q_table:
                q_table[next_obs] = np.zeros(6)

            q_table[obs][action] += alpha * (reward + gamma * np.max(q_table[next_obs]) - q_table[obs][action])
            obs = next_obs  # Update state

        # print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        epsilon = max(epsilon_end, epsilon * decay_rate)

    print("Training Done!")
    print("Q-table:")
    print(q_table)
    print("Saving Q-table...")
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)


action_count = [0, 0, 0, 0]
prev_dir = None
epsilon = 0.995
q_table = {}
is_trained = False  # 追蹤是否已訓練

def get_action(obs):
    global is_trained, q_table, action_count, prev_dir

    if isinstance(obs, dict):
        obs = tuple(obs.values())  # Convert dictionary to tuple

    if not is_trained:
        train()
        is_trained = True
    else:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)

    
    if random.random() < epsilon:
        prob = get_action_probs(action_count, exponent=1.5)
        action = np.random.choice([0, 1, 2, 3], p=prob)
    elif random.random() < epsilon + (1 - epsilon) / 2:
        prev_dir, action = no_prev_action(obs, prev_dir)
    else:
        index = encode_taxi_v3_state(obs)
        if index not in q_table:
            q_table[index] = np.zeros(4)
        action = np.argmax(q_table[index])

    action_count[action] += 1
    return action

