import torch
import yaml
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from PIL import Image
from DQN import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to preprocess game frames
def preprocess_state(state):
    state = Image.fromarray(state).convert('L').resize((84, 84))
    state = np.array(state, dtype=np.float32) / 255.0
    state = np.expand_dims(state, axis=0)  # Add batch dimension
    state = np.expand_dims(state, axis=0)  # Add channel dimension
    return state

def load_agent(model_path, action_size):
    model = DQN(action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def test_agent(model, episodes=10):
    env = gym_tetris.make(config['environment']['game_mode'])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    for episode in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = torch.FloatTensor(state).to(device)
        done = False
        total_reward = 0
        while not done:
            env.render()
            with torch.no_grad():
                # state already has the correct shape, no need to unsqueeze
                action = np.argmax(model(state).cpu().data.numpy())
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            state = torch.FloatTensor(next_state).to(device)
            total_reward += reward
            if done:
                print(f"Test Episode: {episode+1}, Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    action_size = 6  #SIMPLE_MOVEMENT's size is 6
    model_path = 'DQN\models\DQN_episode_10000.pth'
    model = load_agent(model_path, action_size)
    test_agent(model, episodes=10)
