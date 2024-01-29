import json
import os
import pickle
import sys

import neat
import pygame

# Custom modules
import visualize
from Car import Car

# Initializing Pygame
pygame.init()

# Setting up Pygame window
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = pygame.font.Font('freesansbold.ttf', 20)

# Colors
RED = (250, 0, 0)

# Loading goals from a JSON file
json_file_path = "goals1.json"
with open(json_file_path, "r") as file:
    GOALS = json.load(file)

# Loading track image
TRACK_PATH = os.path.join("assets", "track1.png")
TRACK = pygame.image.load(TRACK_PATH)

# List of names for cars

names = ["Eline", "Nick", "Jan", "Lana", "Sam", "Frank", "Fleur", "Senne", "Margot", "Pascale", "Ronny", "Karin",
         "Billie", "Choco", "Nele", "Elise", "Perra", "Alexander", "Stijn", "Michelle", "Valerie", "Hanne", "Maikel",
         "Wouter", "Robin", "Vincent", "Fara", "Lidvine", "Gauthier", "Kobbe", "Stef", "Ivan", "Gilberte", "Dennis",
         "Suzanna", "Toon", "Theums", "Axel", "Lilly", "Emma"]

# Mapping of node indices to their names for neural network visualization
node_names = {-1: '-60', -2: '-30', -3: '0', -4: '30', -5: '60', -6: '150', -7: '210', -8: 'speed',
              0: 'Left', 1: 'Right', 2: 'Drive', 3: 'Brake'}


# Neural Network Visualizer class to display neural networks
class NeuralNetworkVisualizer(neat.reporting.BaseReporter):
    def __init__(self, filename_prefix='neat_NNGraph/neural_net_generation_', show_disabled=True):
        self.filename_prefix = filename_prefix
        self.show_disabled = show_disabled
        self.counter = 0

    def post_evaluate(self, config, population, species_set, best_genome):
        # Visualize the neural network of the best genome each generation
        filename = f"{self.filename_prefix}{self.counter}_best"
        if self.counter % 10 == 0:
            visualize.draw_net(config, best_genome, True, node_names=node_names,
                               filename=filename, show_disabled=self.show_disabled)
        self.counter += 1


# Function to display scores of cars
def score():
    # Sort cars based on their scores in descending order
    sorted_cars = sorted(batch_cars, key=lambda car: car.score, reverse=True)

    # Display the top 5 scores
    for i, car in enumerate(sorted_cars[:5]):
        position = cardict[car]
        text = FONT.render(f'Car {position} (Score: {car.score})', True, (0, 0, 0))
        SCREEN.blit(text, (600, 20 + i * 25))


# Function to remove a car from the batch
def remove(index):
    batch_cars.pop(index)
    batch_ge.pop(index)
    batch_net.pop(index)


# Function to display statistics on the screen
def statistics():
    global cars, game_speed, ge
    text_1 = FONT.render(f'Cars Alive:  {str(len(batch_cars))}', True, (0, 0, 0))
    text_2 = FONT.render(f'Generation:  {pop.generation + 1}', True, (0, 0, 0))
    text_3 = FONT.render(f'Batch:  {batch_id}', True, (0, 0, 0))

    SCREEN.blit(text_1, (50, 20))
    SCREEN.blit(text_2, (50, 50))
    SCREEN.blit(text_3, (50, 80))


# Function to run a batch of cars
def run_batch(batch):
    global batch_cars, batch_ge, batch_net

    batch_cars, batch_ge, batch_net = batch
    run = True
    deathclock = 2000

    clock = pygame.time.Clock()
    while run:
        deathclock -= 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(batch_cars) == 0:
            break

        # Update fitness scores based on car actions
        for i, car in enumerate(batch_cars):
            if car.has_scored():
                batch_ge[i].fitness += 100
            if abs(car.speed) < 1:
                batch_ge[i].fitness -= 2
            elif car.speed > 0:
                batch_ge[i].fitness += 1

            if not car.alive:
                remove(i)

            elif car.deathclock <= 0:
                batch_ge[i].fitness -= 1000
                remove(i)
            elif deathclock <= 0:
                remove(i)

        # Apply neural network outputs to control cars
        for i, car in enumerate(batch_cars):
            output = batch_net[i].activate(car.get_state())
            if output[0] > 0.5:
                car.direction = 1
            elif output[1] > 0.5:
                car.direction = -1
            else:
                car.direction = 0
            if output[2] > 0.5:
                car.move = 1
            elif output[3] > 0.5:
                car.move = -1
            else:
                car.move = 0

        # Update and draw cars on the screen
        for car in batch_cars:
            car.update()
            car.draw(SCREEN, False, False)

        # Display scores and statistics on the screen
        score()
        statistics()
        pygame.display.update()


# Function to evaluate genomes in a generation
def eval_genomes(genomes, config):
    global cars, ge, nets, cardict, batch_id

    cardict = {}
    cars = []
    ge = []
    nets = []
    batch_size = 40
    batch_id = 0
    batches = []
    batched_ge = []
    batched_nets = []
    batched_cars = []

    for genome_id, genome in genomes:

        if batch_id % batch_size == 0:
            if batch_id != 0:
                batches.append([batched_cars, batched_ge, batched_nets])
            batched_ge = []
            batched_nets = []
            batched_cars = []

        car = Car(track_path=TRACK_PATH, goals=GOALS)
        batched_cars.append(car)
        batched_ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        batched_nets.append(net)
        genome.fitness = 0
        cardict[car] = names[batch_id % batch_size]
        batch_id += 1
    batches.append([batched_cars, batched_ge, batched_nets])

    batch_id = 0
    for batch in batches:
        batch_id += 1
        run_batch(batch)


# Main function to set up and run NEAT algorithm
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    checkpointer = neat.Checkpointer(filename_prefix='neatCheckPoint/neat-checkpoint-')

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(checkpointer)
    neural_network_visualizer = NeuralNetworkVisualizer()

    pop.add_reporter(neural_network_visualizer)

    # Run NEAT algorithm for a specified number of generations
    winner = pop.run(eval_genomes, 750)

    # Show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the best genome
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

    # Visualize the best neural network
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Set the path for the NEAT configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
