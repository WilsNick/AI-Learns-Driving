import json
import os
import sys
from Car import Car

import pygame

pygame.init()

json_file_path = "goals1.json"
with open(json_file_path, "r") as file:
    GOALS = json.load(file)

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = pygame.font.Font('freesansbold.ttf', 20)
TRACK_PATH = os.path.join("assets", "track1.png")
TRACK = pygame.image.load(TRACK_PATH)


def main():
    # Run the racing game without AI
    car = Car(track_path=TRACK_PATH, goals=GOALS)

    clock = pygame.time.Clock()

    run = True
    # game loop
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # reset screen
        SCREEN.blit(TRACK, (0, 0))

        # get input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            car.direction = -1
        if keys[pygame.K_RIGHT]:
            car.direction = 1
        if keys[pygame.K_UP]:
            car.move = 1
        elif keys[pygame.K_DOWN]:
            car.move = -1
        else:
            car.move = 0

        # Update the car
        car.update()
        # check if dead
        if not car.alive:
            print("DEAD")
            return True
        # draw the car
        car.draw(SCREEN)

        clock.tick(30)
        pygame.display.update()


if __name__ == '__main__':
    running = True
    while running:
        running = main()
