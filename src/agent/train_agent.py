import os
import numpy as np
from src.environment.game_environment import GameEnvironment
from src.agent.agent import Agent
from src.neural_network.network import build_neural_network

EPISODES = 1000
MODEL_DIR = 'models/'
LAST_MODEL = None

if __name__ == "__main__":
    env = GameEnvironment(driver_path='/path/to/chromedriver')
    state_size = (84, 84, 4)  # Example state size, adjust accordingly
    action_space = 3  # Example action space, adjust accordingly
    agent = Agent(action_space, state_size)
    agent.model = build_neural_network(state_size, action_space, agent.learning_rate)

    # Check if there are saved models and load the most recent one
    if os.path.exists(MODEL_DIR):
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
        if model_files:
            LAST_MODEL = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            agent.load(os.path.join(MODEL_DIR, LAST_MODEL))
            print(f"Loaded model {LAST_MODEL}, resuming training...")

    for e in range(EPISODES):
        state = env.get_screenshot()
        state = np.reshape(state, [1, *state_size])
        done = False
        while not done:
            action = agent.act(state)
            env.press_key(action)  # Adjust mapping for action to key press
            next_state = env.get_screenshot()
            next_state = np.reshape(next_state, [1, *state_size])
            reward = 0  # Adjust reward calculation logic
            done = False  # Adjust logic for determining when the episode is done
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e}/{EPISODES}, score: {reward}")
                break

        if len(agent.memory) > 32:
            agent.replay(32)

        if e % 50 == 0:  # Save model every 50 episodes
            agent.save(f"{MODEL_DIR}super_mario_{e}.h5")

    env.close()
