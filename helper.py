from typing import Union, Tuple

import matplotlib.pyplot as plt
from IPython import display
from Direction import Direction, Point, Slope

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    # plt.pause(.1)


def getScoresFromAgents(agents):
    # each agent should have a list of scores
    # return a list of lists
    scores = []
    for agent in agents:
        scores.append(agent.scores)
    return scores


def getMeanScoresFromAgents(agents):
    # each agent should have a list of scores
    # return a list of lists
    mean_scores = []
    for agent in agents:
        mean_scores.append(agent.mean_scores)
    return mean_scores


def plotAllMean(agents):
    plt.figure(2)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    ax = plt.subplot(111)
    # plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('mean Score')
    for i, agent in enumerate(agents):
        ax.plot(agent.mean_scores, label=agent.name)
        if len(agent.mean_scores) > 0:
            plt.text(len(agent.mean_scores) - 1, agent.mean_scores[-1], str(agent.name))
            plt.text(len(agent.mean_scores) - 1, agent.mean_scores[-1], str(agent.name))
    plt.ylim(ymin=0)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07),
               ncol=len(agents), fancybox=True, shadow=True)
    plt.show(block=False)
    # plt.pause(0.001)


def plotFps(fps):
    plt.figure(1)
    plt.clf()
    plt.title('average fps')
    plt.xlabel('last 100 frames')
    # always have the same height
    plt.ylim(0, 500)
    plt.ylabel('fps')
    mean = sum(fps) / len(fps)
    plt.hlines(mean, 0, len(fps), colors='r', linestyles='dashed')
    plt.hlines(min(fps), 0, len(fps), colors='r', linestyles='dashed')
    plt.hlines(max(fps), 0, len(fps), colors='r', linestyles='dashed')
    plt.plot(fps)
    plt.show(block=False)
    # plt.pause(0.001)


def tint_color(original_color, tint_amount):
    r, g, b = original_color
    tinted_color = (
        min(255, r + tint_amount),
        min(255, g + tint_amount),
        min(255, b + tint_amount)
    )
    return tinted_color

class Vision(object):
    __slots__ = ('dist_to_wall', 'dist_to_apple', 'dist_to_self')
    def __init__(self,
                 dist_to_wall: Union[float, int],
                 dist_to_apple: Union[float, int],
                 dist_to_self: Union[float, int]
                 ):
        self.dist_to_wall = float(dist_to_wall)
        self.dist_to_apple = float(dist_to_apple)
        self.dist_to_self = float(dist_to_self)


### These lines are defined such that facing "up" would be L0 ###
# Create 16 lines to be able to "see" around
VISION_16 = (
    #   L0            L1             L2             L3
    Slope(-1, 0), Slope(-2, 1), Slope(-1, 1), Slope(-1, 2),
    #   L4            L5             L6             L7
    Slope(0, 1), Slope(1, 2), Slope(1, 1), Slope(2, 1),
    #   L8            L9             L10            L11
    Slope(1, 0), Slope(2, -1), Slope(1, -1), Slope(1, -2),
    #   L12           L13            L14            L15
    Slope(0, -1), Slope(-1, -2), Slope(-1, -1), Slope(-2, -1)
)

# Create 8 lines to be able to "see" around
# Really just VISION_16 without odd numbered lines
VISION_8 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i % 2 == 0])

# Create 4 lines to be able to "see" around
# Really just VISION_16 but removing anything not divisible by 4
VISION_4 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i % 4 == 0])
