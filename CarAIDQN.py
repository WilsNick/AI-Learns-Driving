import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import os
from Car import Car
import json

# Define constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_CAPACITY = 10000


# Initialize Pygame
pygame.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = pygame.font.Font('freesansbold.ttf', 20)

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading goals from a JSON file
json_file_path = "goals1.json"
with open(json_file_path, "r") as file:
    GOALS = json.load(file)

# Loading track image
TRACK_PATH = os.path.join("assets", "track1.png")
TRACK = pygame.image.load(TRACK_PATH)

# Define Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Check if running in an IPython environment
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


# Define the RaceEnvironment class
class RaceEnvironment:
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.car = Car(track_path=TRACK_PATH, goals=GOALS)
        self.deadlock = 2000

    def get_score(self):
        return self.car.score

    def reset(self):
        SCREEN.blit(TRACK, (0, 0))
        self.car.reset()

        state = self.car.get_state()
        info = None
        self.car.draw(SCREEN)
        pygame.display.update()

        return state, info

    def step(self, action):
        SCREEN.blit(TRACK, (0, 0))
        observation, reward, terminated = self.car.update_with_action(action, True)

        truncated = self.deadlock <= 0
        info = None
        self.car.draw(SCREEN)
        pygame.display.update()

        return observation, reward, terminated, truncated, info


# Define the ReplayMemory class
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the DQN neural network model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, dropout_prob=0.1):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.layers(x)


# Define the DQNAgent class
class DQNAgent:
    def __init__(self, n_observations, n_actions, device):
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.steps_done = 0

    def select_action(self, state, env_action_space, device):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.sample(env_action_space, 1)[0]]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)


# Function to plot training durations
def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Main code loop
def main():
    env = RaceEnvironment()
    n_actions = len(env.action_space)
    state, info = env.reset()
    n_observations = len(state)

    agent = DQNAgent(n_observations, n_actions, device)

    episode_durations = []

    for i_episode in range(5000):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = agent.select_action(state, env.action_space, device)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.optimize_model()
            agent.update_target_net()

            if done:
                episode_durations.append(env.get_score())
                if i_episode % 50 == 0:
                    plot_durations(episode_durations)

                break

    print('Complete')
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
