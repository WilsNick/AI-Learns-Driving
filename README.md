# AI-Learns-Driving
This project provides a custom racing environment designed for training artificial intelligence agents using NEAT, PPO, and DQN algorithms. The environment allows you to compare the performance of these algorithms.

## Files and Directory Structure

- assets/: Directory containing pictures and JSON files specifying goal placements for AI training.


- Car.py: Contains the implementation of the racing environment.

- Player.py: Runnable file to control the car manually and play the game.

- CarAINeat.py: File for training an AI using NEAT-python.

- CarAIPPO.py: File for training an AI using PPO.

- CarAIDQN.py: File for training an AI using DQN.


- PPO.py: Contains the PPO class.

- visualize.py: Visualizer for NEAT.

- config.txt: Configuration file for NEAT.

- create_walls.py: Used to draw goals and walls for the car environment.

## Getting Started
1. Ensure you have the necessary dependencies installed. You can install them using:

```
pip install -r requirements.txt
```
2. Run the player interface to manually control the car:
```
python Player.py
```
3. Train an AI using NEAT:
```
python CarAINeat.py
```
4. Train an AI using PPO:
```
python CarAIPPO.py
```
5. Train an AI using DQN:
```
python CarAIDQN.py
```
## Additional Information
The assets/ directory contains essential resources for the environment, such as images and JSON files specifying goal placements. Ensure these assets are present and properly configured.

Experiment with different algorithms to observe and compare their performances.

Adjust parameters in the config.txt file to fine-tune the NEAT algorithm.

The collision detection and radars work based on the color of the grass, keep this in mind
when using different tracks.


