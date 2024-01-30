import pygame
import json
import os

# Initialize Pygame
pygame.init()

# Global Constants
HEIGHT = 900
WIDTH = 900
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

TRACK = pygame.image.load(os.path.join("assets", "track1.png"))

# Wall drawing tool
class WallTool:
    def __init__(self):
        self.walls = []
        self.goals = []

    def draw_wall(self, start, end):
        self.walls.append((start, end))

    def draw_goal(self, start, end):
        self.goals.append((start, end))

    def save_walls(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.walls, file)

    def load_walls(self, filename):
        with open(filename, 'r') as file:
            self.walls = json.load(file)
    def save_goals(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.goals, file)

    def load_goals(self, filename):
        with open(filename, 'r') as file:
            self.goals = json.load(file)

# Initialize Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wall Drawing Tool")
clock = pygame.time.Clock()

# Create instances of WallTool
wall_tool = WallTool()

running = True
drawing = False
drawing_goals = False
start_pos = None

while running:
    # screen.fill(WHITE)

    screen.blit(TRACK, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                start_pos = event.pos
            if event.button == 3:  # Right mouse button
                drawing_goals = True
                start_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
                end_pos = event.pos
                wall_tool.draw_wall(start_pos, end_pos)
            if event.button == 3:  # Right mouse button
                drawing_goals = False
                end_pos = event.pos
                wall_tool.draw_goal(start_pos, end_pos)

    if drawing:
        pygame.draw.line(screen, BLACK, start_pos, pygame.mouse.get_pos(), 4)

    if drawing_goals:
        pygame.draw.line(screen, RED, start_pos, pygame.mouse.get_pos(), 4)
    for wall in wall_tool.walls:

        pygame.draw.line(screen, BLACK, wall[0], wall[1], 4)

    for goal in wall_tool.goals:
        pygame.draw.line(screen, RED, goal[0], goal[1], 4)

    pygame.display.flip()
    clock.tick(FPS)

# Save walls to a file
wall_tool.save_walls(os.path.join("assets", "walls3.json"))

# Load walls from the file
wall_tool.load_walls(os.path.join("assets", "walls3.json"))


# Save walls to a file
wall_tool.save_goals(os.path.join("assets", "goals3.json"))

# Load walls from the file
wall_tool.load_goals(os.path.join("assets", "goals3.json"))

pygame.quit()
