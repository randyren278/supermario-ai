import numpy as np
from src.environment.game_environment import GameEnvironment
from src.agent.agent import Agent
from src.neural_network.network import build_neural_network

if __name__ == "__main__":
    env = GameEnvironment(driver_path='/Users/randyren/Developer/chromedriver-mac-arm64/chromedriver')
    state_size = (84, 84, 4)  # Example state size, adjust accordingly
    action_space = 3  # Example action space, adjust accordingly
    agent = Agent(action_space, state_size)
    agent.model = build_neural_network(state_size, action_space, agent.learning_rate)
    agent.load("models/super_mario_latest.h5")  # Load your trained model

    state = env.get_screenshot()
    state = np.reshape(state, [1, *state_size])
    done = False
    while not done:
        action = agent.act(state)
        env.press_key(action)  # Adjust mapping for action to key press
        next_state = env.get_screenshot()
        next_state = np.reshape(next_state, [1, *state_size])
        state = next_state
        done = False  # Adjust logic for determining when the episode is done

    env.close()
