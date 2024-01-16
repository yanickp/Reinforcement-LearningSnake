import pygame
import random
from collections import deque
import helper
from Qlearning_agent_vanilla import Agent_valilla
from Qlearning_agent import QLearningAgent
from Direction import Direction, Point
import concurrent.futures

pygame.init()
font = pygame.font.Font('arial.ttf', 18)
# font = pygame.font.SysFont('arial', 25)


# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20


# SPEED = 4000

class SnakeGameAI:

    def __init__(self, w=1200, h=680, uniqueFood=False, headless=False, speed=4000):
        self.w = w
        self.h = h
        # init display
        self.uniqueFood = uniqueFood
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.timesReset = 0
        self.foodAge = 0
        self.speed = speed
        self.headless = headless

        self.agents = []
        self.fps = deque(maxlen=1000)

        self.reset()

    def reset(self):
        for agent in self.agents:
            agent.reset()

        self._place_food()
        self.timesReset += 1

        self._place_food()
        self.frame_iteration = 0

    def _get_free_spot(self, agent):
        centerX = self.w // 2
        centerY = self.h // 2

        if self.timesReset < 400_000:
            # place the food in a 2 block radius of the head
            dist_from_head = (self.timesReset // 500) + 2 # every 500 resets, increase the distance by 1
            head = agent.snake[0]

            # Generate random offsets within a 2-block radius
            offset_x = random.randint(-dist_from_head, dist_from_head) * BLOCK_SIZE
            offset_y = random.randint(-dist_from_head, dist_from_head) * BLOCK_SIZE

            # Calculate the new coordinates within the grid
            x = (head.x + offset_x) // BLOCK_SIZE * BLOCK_SIZE
            y = (head.y + offset_y) // BLOCK_SIZE * BLOCK_SIZE
        else:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        # print("food at: ", x, y)

        # if food is in any of the snakes
        for agent in self.agents:
            if Point(x, y) in agent.snake:
                self._place_food()

        return Point(x, y)

    def _place_food(self):
        if self.uniqueFood:
            for agent in self.agents:
                if agent.food is None:
                    agent.food = self._get_free_spot(agent=agent)
                    # agent.food = Point(600, 680)
        else:
            freeSpot = self._get_free_spot()
            for agent in self.agents:
                agent.food = freeSpot

    def play_step(self, action, agent):
        self.frame_iteration += 1
        self.foodAge += 1
        agent.TimeNotEaten += 1
        reward = 0
        if action == agent.previous_action:
            agent.repeated_count += 1
            # print("Agent " + str(agent.name) + " repeated action " + str(agent.repeated_count) + " times")
        else:
            agent.repeated_count = 0

        agent.previous_action = action
        repetative_penalty = -0.5 * agent.repeated_count
        reward += repetative_penalty

        if agent.TimeNotEaten > 50 * len(agent.snake):  # and self.timesReset < 300:
            # print("Agent " + str(agent.name) + " died of starvation")
            return -10, True, agent.score

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        agent._move(action)  # update the head
        agent.snake.insert(0, agent.head)
        agent.TimeNotEaten += 1

        # 3. check if game over
        # reward = 0
        game_over = False
        if agent.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, agent.score

        # 4. place new food or just move
        if agent.head == agent.food:
            agent.score += 1
            agent.TimeNotEaten = 0
            reward = 10
            agent.food = None
            self._place_food()
        else:
            # reward = -(self._distance_to_food() / 10)
            reward = 5 / agent.get_distance_to_food()
            agent.snake.pop()

        # 5. update ui and clock
        if not self.headless:
            self._update_ui()
            self.clock.tick(self.speed)
        # 6. return game over and score
        return reward, game_over, agent.score

    def _update_ui(self):
        self.display.fill(BLACK)

        for index, agent in enumerate(self.agents):
            # draw snake
            # for point, color in agent.drawable_visions: # draw vision lines
            #     pointX, pointY = point
            #     pointX += BLOCK_SIZE / 2
            #     pointY += BLOCK_SIZE / 2
            #     pygame.draw.line(self.display, color, (agent.head.x + BLOCK_SIZE / 2, agent.head.y + BLOCK_SIZE / 2),
            #                      (pointX, pointY))  # left
            # pygame.draw.line(self.display, agent.color, agent.head, (agent.head.x, 0))  # top
            # pygame.draw.line(self.display, agent.color, agent.head, (self.w, agent.head.y))  # right
            # pygame.draw.line(self.display, agent.color, agent.head, (agent.head.x, self.h))  # bottom
            # agent.get_distance_from_head_to(Point(0, agent.head.y))
            for pt in agent.snake:
                pygame.draw.rect(self.display, agent.color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, agent.accent_color, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

            text = font.render(str(agent.name) + "'s score: " + str(agent.score) + " hs: " + str(agent.record), True,
                               agent.color)

            self.display.blit(text, [0, 20 * (index + 1)])
        self.display.blit(
            font.render("game: " + str(self.timesReset) + " fps: " + str(round(self.clock.get_fps(), 2)), True, WHITE),
            [0, 0])

        for agent in self.agents:
            if agent.food is not None:
                pygame.draw.rect(self.display, agent.color,
                                 pygame.Rect(agent.food.x, agent.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # pygame.draw.line(self.display, RED, (self.head.x, self.head.y), (self.food.x, self.food.y), 2)
        # draw rectangle for posible food locations
        self.fps.append(self.clock.get_fps())  # add to fps list max len 1000
        pygame.display.flip()

    def printScores(self):
        for agent in self.agents:
            print("Agent " + str(agent.name) + " score: " + str(agent.score) + " record: " + str(
                agent.record) + " avg: " + str(agent.total_score / agent.n_games))

    def addAgent(self, name):
        self.agents.append(Agent_valilla(self.w, self.h, BLOCK_SIZE, name))

    def addAgents(self, amount):
        for i in range(len(self.agents), len(self.agents) + amount, 1):
            self.agents.append(Agent_valilla(self.w, self.h, BLOCK_SIZE, str("agent" + str(i))))

    def addDeepQagent(self, name, layers=[128, 128]):
        self.agents.append(QLearningAgent(self.w, self.h, BLOCK_SIZE, name=name, layers=layers))


def runTraining():
    game = SnakeGameAI(w=400, h=400, uniqueFood=True,
                       headless=False,
                       speed=10000000)  # uniqueFood means if they all fight for the same food or not, headless means no UI so faster training
    gameoverCount = 0

    # test = QLearningAgent(game.w, game.h, BLOCK_SIZE, name="vis2012New", layers=[20, 12], inputSize=20)
    # test1 = QLearningAgent(game.w, game.h, BLOCK_SIZE, name="vis128New", layers=[128], inputSize=20)
    # test1 = QLearningAgent(game.w, game.h, BLOCK_SIZE, name="Tail", layers=[256], inputSize=15)
    # test1.tailInfo = True
    # test1 = QLearningAgent(game.w, game.h, BLOCK_SIZE, name="new vis", layers=[8, 3], inputSize=26)
    # test2 = QLearningAgent(game.w, game.h, BLOCK_SIZE, name="new vis2", layers=[64], inputSize=26)
    # test3 = QLearningAgent(game.w, game.h, BLOCK_SIZE, name="new vis3", layers=[128], inputSize=26)
    test4 = QLearningAgent(game.w, game.h, BLOCK_SIZE, name="new vis4", layers=[256], inputSize=26)

    # game.addAgents(1)
    # test.LR = 0.01
    # test = Agent_valilla(game.w, game.h, BLOCK_SIZE, name="valillaVision")
    # game.agents.append(test1)
    # game.agents.append(test2)
    # game.agents.append(test3)
    game.agents.append(test4)


    game.reset() # start the game
    while True:
        gameoverCount = 0
        # game._update_ui()
        for agent in game.agents:
            if not agent.isDead:
                state = agent.get_state()
                action = agent.get_action(state)
                reward, game_over, score = game.play_step(action, agent)
                nextState = agent.get_state()
                agent.train(state, action, reward, nextState, game_over, score)
                if game_over:
                    gameoverCount += 1
            else:
                gameoverCount += 1

            if gameoverCount >= len(game.agents):
                # game.printScores()
                gameoverCount = 0
                if game.timesReset % 1 == 0:
                    helper.plotAllMean(game.agents)
                # helper.plotFps(game.fps)
                game.reset()


if __name__ == '__main__':
    runTraining()
