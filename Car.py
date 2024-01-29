import json
import math
import os

import pygame


TRACK = pygame.image.load(os.path.join("assets", "track1.png"))
RED = (250, 0, 0)


class Track:
    def __init__(self, track_image_path):
        self.track_image = pygame.image.load(track_image_path)

    def get_width(self):
        return self.track_image.get_width()

    def get_height(self):
        return self.track_image.get_height()

    def is_collision(self, x, y):
        try:
            # Get pixel color at the specified coordinates
            pixel_color = self.track_image.get_at((int(x), int(y)))
            # Define the track color (adjust as needed)
            track_color = pygame.Color(178, 219, 160, 255)
            return pixel_color == track_color
        except IndexError:
            # Handle out-of-bounds access as a collision
            return True


class Car:
    def __init__(self, track_path, goals):
        # Load images and goals
        self.original_image = pygame.image.load(os.path.join("assets", "car.png"))
        self.original_image = pygame.transform.scale(self.original_image, (
            int(self.original_image.get_width() * 0.1), int(self.original_image.get_height() * 0.1)))
        self.image = self.original_image
        self.track = Track(track_path)
        self.goals = goals

        # self.track = Track(os.path.join("assets", "track1.png"))

        # Initial car position and properties
        self.X_POS = 120
        self.Y_POS = 800
        self.rect = self.image.get_rect(center=(150, 800))
        self.angle = 0
        self.direction_to_go = (1, 0)
        self.difference_rotation = (0, 0)
        self.center_screen = (self.track.get_width() / 2, self.track.get_height() / 2)
        self.blit_rotate(self.image, self.angle)  # Rotate by 10 degrees

        # Car controls and state
        self.acceleration = 0.5
        self.speed = 0
        self.max_speed = 20
        self.angle_speed = 6
        self.direction = 0
        self.move = 0
        self.alive = True
        self.radars = [[-120, 0], [-60, 0], [-30, 0], [0, 0], [30, 0], [60, 0], [120, 0]]

        # Time and score-related variables
        self.max_time = 100
        self.deathclock = self.max_time
        self.score = 0
        self.current_goal = 0
        self.scored = False

    def reset(self):
        # Reset car to initial state
        self.X_POS = 120
        self.Y_POS = 800
        self.alive = True
        self.rect = self.image.get_rect(center=(150, 800))
        self.angle = 0
        self.direction_to_go = (1, 0)

        self.radars = [[-120, 0], [-60, 0], [-30, 0], [0, 0], [30, 0], [60, 0], [120, 0]]
        self.score = 0
        self.current_goal = 0
        self.move = 0
        self.direction = 0
        self.speed = 0
        self.deathclock = self.max_time
        self.score = 0
        self.scored = False

    def get_state(self):
        # Get distances from radars (normalized between 0 and 1)
        max_radar_distance = 200
        normalized_radars = [dist / max_radar_distance for _, dist in self.radars]

        # Normalize speed to the range [0, 1]
        normalized_speed = (self.speed + self.max_speed) / (2 * self.max_speed)

        # Combine all features into a state representation
        state = normalized_radars + [normalized_speed]

        return state

    def update_with_action(self, action, timed=False):
        # Update car state based on the selected action
        # Timed tells whether you want the information of the deathclock
        if action == 0:  # Steer left
            self.direction = -1
            self.move = 0
        elif action == 1:  # Steer right
            self.direction = 1
            self.move = 0
        elif action == 2:  # Steer up
            self.direction = 0
            self.move = 1
        elif action == 3:  # Steer down
            self.direction = 0
            self.move = -1
        elif action == 4:  # Steer left up
            self.direction = -1
            self.move = 1
        elif action == 5:  # Steer right up
            self.direction = 1
            self.move = 1
        elif action == 6:  # Steer left down
            self.direction = -1
            self.move = -1
        elif action == 7:  # Steer right down
            self.direction = 1
            self.move = -1
        elif action == 8:  # do nothing
            self.direction = 0
            self.move = 0

        self.update()
        reward = self.calculate_reward()
        if timed:
            return self.get_state(), reward, not self.alive or self.deathclock <= 0
        return self.get_state(), reward, not self.alive

    def calculate_reward(self):
        # Calculate the reward of the current state
        reward = 0
        if self.has_scored():
            reward += 100
        if self.alive and self.speed > 0:
            reward += 1
        else:
            reward = -10
        return reward

    def blit_rotate(self, image, angle):
        # Rotate the car to the given angle
        self.rotated_image = pygame.transform.rotate(image, angle)
        self.rect = self.rotated_image.get_rect(center=image.get_rect(topleft=self.center_screen).center)
        self.og_rect = self.original_image.get_rect(center=image.get_rect(topleft=self.center_screen).center)
        self.difference_rotation = (self.rect.x - self.og_rect.x, self.rect.y - self.og_rect.y)

    def change_direction(self, angle):
        # Change the direction where the car points too
        angle_radians = math.radians(angle)
        new_direction = (math.cos(angle_radians), math.sin(angle_radians))
        return new_direction

    def update(self):
        # update the state of the car

        # account for the rotation of the car
        self.rect.x -= self.difference_rotation[0]
        self.rect.y -= self.difference_rotation[1]
        # move forward
        self.drive()
        # steer left or right
        self.rotate()
        self.direction_to_go = self.change_direction(-self.angle)

        # account for the rotation
        self.X_POS += self.speed * self.direction_to_go[0]
        self.Y_POS += self.speed * self.direction_to_go[1]
        # update the position of the car
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.rect.x += self.difference_rotation[0]
        self.rect.y += self.difference_rotation[1]

        # update sensor and collisions
        self.sense_radar()
        self.collision()
        # update score
        self.check_score()
        self.deathclock -= 1

    def drive(self):
        self.speed += self.move * self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -(self.max_speed / 2):
            self.speed = -self.max_speed / 2
        if self.move == 0:
            if self.speed > 1:
                self.speed -= self.acceleration / 10
            elif self.speed < -1:
                self.speed += self.acceleration / 10
            else:
                self.speed = 0

    def collision(self):
        try:
            length = 22
            collision_point_right = [self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length,
                                     self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length]
            collision_point_left = [self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length,
                                    self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length]
            collision_point_bottom_right = [self.rect.center[0] - math.cos(math.radians(self.angle - 18)) * length,
                                            self.rect.center[1] + math.sin(math.radians(self.angle - 18)) * length]
            collision_point_bottom_left = [self.rect.center[0] - math.cos(math.radians(self.angle + 18)) * length,
                                           self.rect.center[1] + math.sin(math.radians(self.angle + 18)) * length]

            if self.track.is_collision(*collision_point_right) or self.track.is_collision(*collision_point_left):
                self.alive = False

            if self.track.is_collision(*collision_point_bottom_right) or self.track.is_collision(
                    *collision_point_bottom_left):
                self.alive = False
        except Exception as varname:
            self.alive = False

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.angle_speed
        if self.direction == -1:
            self.angle += self.angle_speed
        self.direction = 0

        self.blit_rotate(self.original_image, self.angle)
        self.image = self.rotated_image

    # Modify the radar method in your Car class:
    def sense_radar(self):

        for i, radar in enumerate(self.radars):
            length = 0
            x = int(self.rect.center[0])
            y = int(self.rect.center[1])

            try:
                while not self.track.is_collision(x, y) and length < 200:
                    length += 1
                    x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar[0])) * length)
                    y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar[0])) * length)

                dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
                self.radars[i][1] = dist
            except Exception as varname:
                self.alive = False
                self.radars[i][1] = 0

    def has_scored(self):
        # Check if the car has just scored a goal
        if self.scored:
            self.scored = False
            return True
        return False

    def check_score(self):
        # check if the car passed a goal
        car_rect = self.rect
        line_start = self.goals[self.current_goal][0]
        line_end = self.goals[self.current_goal][1]

        line_rect = pygame.Rect(line_start, (line_end[0] - line_start[0], line_end[1] - line_start[1]))

        if car_rect.colliderect(line_rect):
            self.score += 1
            self.scored = True
            self.current_goal += 1
            if self.current_goal >= len(self.goals):
                self.current_goal = 0
            self.deathclock = self.max_time
            return

    def draw(self, SCREEN, draw_radar=True, draw_collider=True):
        # Draw the car with the radars and colliders based on the arguments
        # draw the car
        SCREEN.blit(self.image, self.rect)
        # draw the goal
        pygame.draw.line(SCREEN, RED, self.goals[self.current_goal][0], self.goals[self.current_goal][1], 2)
        # draw the radars
        if draw_radar:
            for radar in self.radars:
                radar_angle, length = radar
                x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
                y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

                pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
                pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        # draw the colliders
        if draw_collider:
            length2 = 22
            collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length2),
                                     int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length2)]
            collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length2),
                                    int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length2)]
            collision_point_bottom_right = [
                int(self.rect.center[0] - math.cos(math.radians(self.angle - 18)) * length2),
                int(self.rect.center[1] + math.sin(math.radians(self.angle - 18)) * length2)]
            collision_point_bottom_left = [int(self.rect.center[0] - math.cos(math.radians(self.angle + 18)) * length2),
                                           int(self.rect.center[1] + math.sin(math.radians(self.angle + 18)) * length2)]
            pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
            pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)
            pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_bottom_right, 4)
            pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_bottom_left, 4)
