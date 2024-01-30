# Code based on https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master

import json
import os
import sys
from datetime import datetime

import pygame

from Car import Car
from PPO import PPO

# Initialize Pygame
pygame.init()

# Set up the screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = pygame.font.Font('freesansbold.ttf', 20)

# Loading goals from a JSON file
json_file_path = "goals1.json"
with open(json_file_path, "r") as file:
    GOALS = json.load(file)

# Loading track image
TRACK_PATH = os.path.join("assets", "track1.png")
TRACK = pygame.image.load(TRACK_PATH)


# Define the RaceEnvironment class
class RaceEnvironment:
    def __init__(self):
        # Define the action space and initialize the Car object with track and goals
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.car = Car(track_path=TRACK_PATH, goals=GOALS)
        self.deadlock = 2000  # Maximum deadlock value
        self.env_name = "Car Racing"

    def get_score(self):
        return self.car.score

    def reset(self):
        # Reset the environment and return the initial state
        SCREEN.blit(TRACK, (0, 0))
        self.car.reset()
        state = self.car.get_state()
        self.car.draw(SCREEN)
        pygame.display.update()
        return state

    def step(self, action):
        # Take a step in the environment based on the given action
        SCREEN.blit(TRACK, (0, 0))
        observation, reward, terminated = self.car.update_with_action(action, True)
        truncated = self.deadlock <= 0
        info = None
        self.car.draw(SCREEN)
        pygame.display.update()
        return observation, reward, terminated, info


# Define the PPOTrainer class
class PPOTrainer:
    def __init__(self, env, max_training_timesteps=3e6, max_ep_len=2000, print_freq=20000, log_freq=4000,
                 save_model_freq=1e5, action_std=0.6, action_std_decay_rate=0.05, min_action_std=0.1,
                 action_std_decay_freq=2.5e5, update_timestep=8000, K_epochs=80, eps_clip=0.2, gamma=0.99,
                 lr_actor=0.0003, lr_critic=0.001, random_seed=0, has_continuous_action_space=False):
        # Initialize the PPO trainer with hyperparameters
        self.env = env
        self.max_training_timesteps = max_training_timesteps
        self.max_ep_len = max_ep_len
        self.print_freq = print_freq
        self.log_freq = log_freq
        self.save_model_freq = save_model_freq
        self.action_std = action_std
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.action_std_decay_freq = action_std_decay_freq
        self.update_timestep = update_timestep
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.random_seed = random_seed
        self.start_time = None
        self.env_name = "Car Racing"
        self.has_continuous_action_space = has_continuous_action_space

        self.state_dim = len(self.env.reset())

        if isinstance(self.env.action_space, list):
            self.action_dim = len(self.env.action_space)
        else:
            self.action_dim = self.env.action_space.shape[0]

        self.log_f = None
        self.log_f_name = None
        self.checkpoint_path = None
        self.ppo_agent = None

    def initialize_logging(self):
        # Initialize logging for tracking training progress
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = os.path.join(log_dir, self.env_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        self.log_f_name = os.path.join(log_dir, f'PPO_{self.env_name}_log_{run_num}.csv')

        print("current logging run number for " + self.env_name + " : ", run_num)
        print("logging at : " + self.log_f_name)

    def initialize_checkpointing(self):
        # Initialize checkpointing for saving model checkpoints
        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = os.path.join(directory, self.env_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.checkpoint_path = os.path.join(directory, f'PPO_{self.env_name}_{self.random_seed}.pth')
        print("save checkpoint path : " + self.checkpoint_path)

    def initialize_ppo_agent(self):
        # Initialize the PPO agent for training
        self.ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma,
                             self.K_epochs, self.eps_clip, self.has_continuous_action_space, self.action_std)

    def train(self):
        # Main training loop
        print("============================================================================================")

        self.initialize_logging()
        self.initialize_checkpointing()
        self.initialize_ppo_agent()

        self.start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)

        print("============================================================================================")

        self.log_f = open(self.log_f_name, "w+")
        self.log_f.write('episode,timestep,reward\n')

        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        while time_step <= self.max_training_timesteps:
            state = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.max_ep_len + 1):
                action = self.ppo_agent.select_action(state)
                state, reward, done, _ = self.env.step(action)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                if time_step % self.update_timestep == 0:
                    self.ppo_agent.update()

                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    self.ppo_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                if time_step % self.log_freq == 0:
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    self.log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    self.log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                if time_step % self.print_freq == 0:
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                            print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                if time_step % self.save_model_freq == 0:
                    print(
                        "--------------------------------------------------------------------------------------------")
                    print("saving model at : " + self.checkpoint_path)
                    self.ppo_agent.save(self.checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - self.start_time)
                    print(
                        "--------------------------------------------------------------------------------------------")

                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        self.log_f.close()

        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - self.start_time)
        print("============================================================================================")


if __name__ == '__main__':
    race_env = RaceEnvironment()
    trainer = PPOTrainer(race_env)
    trainer.train()
