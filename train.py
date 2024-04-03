import numpy as np
import yaml
import torch
from PIL import Image
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT, MOVEMENT
import logging
import os
from datetime import datetime
# Models
from DQN_agent import DQN_Agent
from PG import PG_Agent

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to preprocess game frames
def preprocess_state(state):
    state = Image.fromarray(state).convert('L').resize((84, 84))
    state = np.array(state, dtype=np.float32) / 255.0
    state = np.expand_dims(state, axis=0)  # Add batch dimension
    state = np.expand_dims(state, axis=0)  # Add channel dimension
    return state

def save_model(model, filename="tetris_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}.")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    log_folder = config['training']['model_name'] + '/' + config['training']['log_file_folder']
    model_folder = config['training']['model_name']+ '/' + config['training']['model_path_folder']
    ensure_dir(log_folder)
    ensure_dir(model_folder)

    log_file_with_timestamp = log_folder + 'training_log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
    logging.basicConfig(filename=log_file_with_timestamp, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('Training start')

    # Check for CUDA availability and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training loop setup
    env = gym_tetris.make(config['environment']['game_mode'])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    action_size = env.action_space.n

    if config['training']['model_name'] == 'DQN':
         # Assuming this is accessible
        agent = DQN_Agent(action_size)

        batch_size = config['training']['batch_size']
        episodes = config['training']['episodes']

        for e in range(episodes):
            state = env.reset()
            state = preprocess_state(state)
            state = torch.FloatTensor(state).to(device)  # Convert and move to device here
            done = False
            total_reward = 0
            losses = []  # List to track losses for each step

            while not done:
                # env.render()
                action = agent.act(state)  # `state` is already a FloatTensor on the correct device
                next_state, reward, done, _ = env.step(action)
                next_state = preprocess_state(next_state)
                next_state = torch.FloatTensor(next_state).to(device)  # Convert and move to device

                loss = agent.remember(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)  # Store numpy arrays in memory
                if loss is not None:
                    losses.append(loss.item())

                state = next_state
                total_reward += reward

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if loss is not None:  # If replay returned a loss
                    losses.append(loss.item())

            average_loss = np.mean(losses) if losses else 0
            logging.info(f'Episode: {e+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}, Average Loss: {average_loss:.4f}')


            if (e + 1) % config['training']['model_save_interval'] == 0:
                # save_model(agent.model, filename=f"tetris_model_episode_{e+1}.pth")
                save_model(agent.model, filename=model_folder + config['training']['model_name'] + f'_episode_{e+1}.pth')

    elif config['training']['model_name'] == 'PG':
        # Adjust the dimensions according to your preprocessed state and action space
        state_size = (1, 84, 84)  # Example dimensions after preprocessing
        agent = PG_Agent(np.prod(state_size), action_size, device)

        episodes = config['training']['episodes']

        for e in range(episodes):
            state = env.reset()
            state = preprocess_state(state)
            state = np.reshape(state, [1, 84, 84])  # Ensure the state shape matches the expected input
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(state)  # Select action based on the current policy
                next_state, reward, done, _ = env.step(action)
                next_state = preprocess_state(next_state)
                next_state = np.reshape(next_state, [1, 84, 84])

                agent.rewards.append(reward)  # Store reward for policy update
                state = next_state
                total_reward += reward

            agent.update_policy()  # Update the policy at the end of the episode

            logging.info(f'Episode: {e+1}, Total Reward: {total_reward}')

            if (e + 1) % config['training']['model_save_interval'] == 0:
                save_model(agent.model, filename=model_folder + config['training']['model_name'] + f'_episode_{e+1}.pth')

    else:
        raise ValueError("Invalid model name in config file.")