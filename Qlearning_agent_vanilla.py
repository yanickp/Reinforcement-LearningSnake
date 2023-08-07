import pickle
import random
import numpy as np
from Direction import Direction, Point
import threading
from helper import tint_color
import agentParent as AgentParent



class Agent_valilla(AgentParent.agent):

    def __init__(self, board_width, board_height, block_size, name="vanilla", LR=0.1, gamma=0.99):
        super().__init__(board_width, board_height, block_size, name)
        self.q_table = {}
        self.LR = LR
        self.gamma = gamma

    def _get_q_value(self, state, action):
        # Get the Q-value for a specific state-action pair from the Q-table
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # Initialize Q-values for each action in the state
        action_index = np.argmax(action)
        return self.q_table[state][action_index]

    def _update_q_value(self, state, action, new_q_value):
        # Update the Q-value for a specific state-action pair in the Q-table
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]

        action_index = np.argmax(action)
        self.q_table[state][action_index] = new_q_value

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if random.random() < max(0.0, 1 - self.n_games / self.epsilon) and not self.loadedModel:  # or change < 5:
            move = random.randint(0, 2)
        else:
            state = tuple(state)
            move = np.argmax(self.q_table.get(state, [0, 0, 0]))  # Choose the action with the highest Q-value
        action_vector = [0, 0, 0]
        action_vector[move] = 1
        return action_vector

    def loadBrain(self, filename):
        with open(filename, 'rb') as brain:
            self.q_table = pickle.load(brain)
        self.loadedModel = True

    def train(self, state, action, reward, next_state, done, score):
        if not self.isDead:  # If the snake is dead, there is no need to train the Q-table
            state = tuple(state)
            next_state = tuple(next_state)

            current_q = self._get_q_value(state, action)
            max_next_q = max(
                self.q_table.get(next_state,
                                 [0, 0, 0]))  # [0,0,0] is the default value if next_state is not in the Q-table
            new_q = current_q + self.LR * (
                    reward + self.gamma * max_next_q - current_q)  # Learning rate = 0.1, discount factor = 0.99
            self._update_q_value(state, action, new_q)

        if done:
            if score > self.record:
                self.record = score
                with open(f'model/{self.name}H{str(self.record)}.pkl', 'wb') as fp:
                    pickle.dump(self.q_table, fp)

            # game.reset()
            self.isDead = True
            self.n_games += 1
            self.total_score += score
            mean_score = self.total_score / self.n_games
            self.scores.append(score)
            self.mean_scores.append(np.mean(self.scores[-50:]))
            # print(str(self.name) + 's Game', self.n_games, 'Score', score, 'Record:', self.record, 'Mean Score:', mean_score)


if __name__ == '__main__':
    agent = Agent_valilla()
    agent.train()
