import copy
import math
from typing import List

import numpy as np

from Direction import Direction, Point, Slope
import threading
from helper import tint_color, Vision, VISION_8, VISION_4, VISION_16
import random

BLOCK_SIZE = 20


class agent:
    def __init__(self, board_width, board_height, block_size, name):
        self.n_games = 0  # amount of games played

        self.epsilon = 1500  # randomness
        self.record = 0  # highest score
        self.total_score = 0  # overall score
        self.score = 0  # per game

        self.food = None

        self.name = name
        self.isDead = False
        self.TimeNotEaten = 0

        self.loadedModel = False

        self.previous_action = None
        self.repeated_count = 0
        self.sees_food = False

        self.scores = []
        self.mean_scores = []

        self._vision_type = VISION_4
        self._vision: List[Vision] = [None] * len(self._vision_type)
        self.vision_as_array = [0] * len(self._vision_type) * 3
        self.apple_and_self_vision = 'binary'

        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.accent_color = tint_color(self.color, 50)  # tint by 50 for accent color

        self.board_width = board_width
        self.board_height = board_height
        self.BLOCK_SIZE = block_size

        # inittialize the snake
        self.direction = Direction.RIGHT
        self.tailInfo = False
        self.tailDirection = Direction.RIGHT

        self.head = Point(self.board_width / 2, self.board_height / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y)]

        # self.drawable_visions = [Point(0, 0)] * len(self._vision_type)
        self.drawable_visions = []  # List of tuples with (point, color)

    def reset(self):
        self.direction = Direction.RIGHT
        self.tailDirection = Direction.RIGHT
        self.isDead = False
        self.TimeNotEaten = 0
        self.head = Point(self.board_width / 2, self.board_height / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._vision: List[Vision] = [None] * len(self._vision_type)
        self.vision_as_array = [0] * len(self._vision_type) * 3

    def is_collision(self, pt=None) -> bool:
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.board_width - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.board_height - self.BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _wall_collision(self, pt):
        # hits boundary
        if pt.x > self.board_width - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.board_height - self.BLOCK_SIZE or pt.y < 0:
            return True

        return False

    # def get_state(self):
    #     head = self.snake[0]
    #     point_l = Point(head.x - 20, head.y)
    #     point_r = Point(head.x + 20, head.y)
    #     point_u = Point(head.x, head.y - 20)
    #     point_d = Point(head.x, head.y + 20)
    #
    #     dir_l = self.direction == Direction.LEFT
    #     dir_r = self.direction == Direction.RIGHT
    #     dir_u = self.direction == Direction.UP
    #     dir_d = self.direction == Direction.DOWN
    #
    #     if self.food is not None:
    #         food_left = self.food.x < head.x  # food left
    #         food_right = self.food.x > head.x  # food right
    #         food_up = self.food.y < head.y  # food up
    #         food_down = self.food.y > head.y  # food down
    #     else:
    #         food_left = False
    #         food_right = False
    #         food_up = False
    #         food_down = False
    #
    #     state = [
    #         # Danger straight
    #         (dir_r and self.is_collision(point_r)) or
    #         (dir_l and self.is_collision(point_l)) or
    #         (dir_u and self.is_collision(point_u)) or
    #         (dir_d and self.is_collision(point_d)),
    #
    #         # Danger right
    #         (dir_u and self.is_collision(point_r)) or
    #         (dir_d and self.is_collision(point_l)) or
    #         (dir_l and self.is_collision(point_u)) or
    #         (dir_r and self.is_collision(point_d)),
    #
    #         # Danger left
    #         (dir_d and self.is_collision(point_r)) or
    #         (dir_u and self.is_collision(point_l)) or
    #         (dir_r and self.is_collision(point_u)) or
    #         (dir_l and self.is_collision(point_d)),
    #
    #         # Move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
    #
    #         # Food location
    #         food_left,
    #         food_right,
    #         food_up,
    #         food_down,
    #     ]
    #
    #     if self.tailInfo:
    #         state.append(self.tailDirection == Direction.LEFT)
    #         state.append(self.tailDirection == Direction.RIGHT)
    #         state.append(self.tailDirection == Direction.UP)
    #         state.append(self.tailDirection == Direction.DOWN)
    #
    #     return np.array(state, dtype=int)

    def get_state(self):
        state = []
        tailDR = self.get_tail_direction()
        self.look()
        for i, value in enumerate(self.vision_as_array):
            state.append(self.vision_as_array[i])
        # tail dirrection
        state.append(tailDR == Direction.LEFT)
        state.append(tailDR == Direction.RIGHT)
        state.append(tailDR == Direction.UP)
        state.append(tailDR == Direction.DOWN)
        # snake direction
        state.append(self.direction == Direction.LEFT)
        state.append(self.direction == Direction.RIGHT)
        state.append(self.direction == Direction.UP)
        state.append(self.direction == Direction.DOWN)
        return np.array(state, dtype=float)

    def _move(self, action):
        # [straight, right, left]
        if not self.isDead:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

            self.direction = new_dir
            self.tailDirection = self.get_tail_direction()
            # print('tailDirection: ' + str(self.tailDirection))

            x = self.head.x
            y = self.head.y
            if self.direction == Direction.RIGHT:
                x += self.BLOCK_SIZE
            elif self.direction == Direction.LEFT:
                x -= self.BLOCK_SIZE
            elif self.direction == Direction.DOWN:
                y += self.BLOCK_SIZE
            elif self.direction == Direction.UP:
                y -= self.BLOCK_SIZE

            self.head = Point(x, y)

    def get_tail_direction(self):
        p2 = self.snake[-2]
        p1 = self.snake[-1]
        diffx = p2.x - p1.x
        diffy = p2.y - p1.y
        if diffx < 0:
            return Direction.LEFT
        elif diffx > 0:
            return Direction.RIGHT
        elif diffy > 0:
            return Direction.DOWN
        elif diffy < 0:
            return Direction.UP

    def isInBody(self, point) -> bool:
        for body_part in self.snake:
            # skip head
            if body_part == self.head:
                continue
            if body_part == point:
                return True
        return False

    def _vision_as_input_array(self) -> None:
        # Update the input array
        # Split _vision into np array where rows [0-2] are _vision[0].dist_to_wall, _vision[0].dist_to_apple, _vision[0].dist_to_self,
        # rows [3-5] are _vision[1].dist_to_wall, _vision[1].dist_to_apple, _vision[1].dist_to_self, etc. etc. etc.
        for i, value in enumerate(self._vision):
            self.vision_as_array[i * 3] = value.dist_to_wall
            self.vision_as_array[i * 3 + 1] = value.dist_to_apple
            self.vision_as_array[i * 3 + 2] = value.dist_to_self
        # normalize the self.vision_as_array
        # self.vision_as_array = (self.vision_as_array - np.mean(self.vision_as_array)) / np.std(self.vision_as_array)

    def look(self):
        # Look all around
        self.drawable_visions = []
        self.sees_food = False
        for i, slope in enumerate(self._vision_type):
            vision = self.look_in_direction(slope)
            if vision.dist_to_apple < np.inf and self.apple_and_self_vision != 'binary':
                self.sees_food = True
            if vision.dist_to_apple == 1 and self.apple_and_self_vision == 'binary':
                self.sees_food = True
            self._vision[i] = vision

        # Update the input array
        self._vision_as_input_array()

    def look_in_direction(self, slope: Slope):
        dist_to_wall = None
        dist_to_apple = np.inf
        dist_to_self = np.inf

        wall_location = None
        apple_location = None
        self_location = None

        # position = self.snake[0].copy()
        position = Point(int(self.head.x), int(self.head.y))
        distance = 1.0
        total_distance = 0.0

        # Can't start by looking at yourself
        position.x += int(slope.horizontal) * self.BLOCK_SIZE
        position.y += int(slope.vertical) * self.BLOCK_SIZE
        total_distance += distance
        body_found = False  # Only need to find the first occurance since it's the closest
        food_found = False
        while not self._wall_collision(position):
            # distance_to_food = math.sqrt((self.food.x - position.x) ** 2 + (self.food.y - position.y) ** 2)
            if not body_found and self.isInBody(position):
                body_found = True
                self_location = position
                dist_to_self = total_distance
            if not food_found and self.food is not None and position == self.food:
                food_found = True
                dist_to_apple = total_distance
                apple_location = position

            position.x += int(slope.horizontal) * self.BLOCK_SIZE
            position.y += int(slope.vertical) * self.BLOCK_SIZE
            total_distance += distance
        assert (total_distance != 0.0)

        # dist_to_wall = 1 / total_distance
        # normalize distance to wall between 0 and 1
        dist_to_wall = total_distance / (self.board_width / self.BLOCK_SIZE)
        # dist_to_wall = 1.0 / dist_to_wall

        if self.apple_and_self_vision == 'binary':
            dist_to_apple = 1.0 if dist_to_apple != np.inf else 0.0
            dist_to_self = 1.0 if dist_to_self != np.inf else 0.0

        elif self.apple_and_self_vision == 'distance':
            dist_to_apple = 1.0 / dist_to_apple
            dist_to_self = 1.0 / dist_to_self

        vision = Vision(dist_to_wall, dist_to_apple, dist_to_self)
        new_point = (position.x, position.y)
        self.drawable_visions.append((new_point, (255 * dist_to_self, dist_to_apple * 255, 100)))
        return vision
