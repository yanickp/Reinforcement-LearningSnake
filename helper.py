import matplotlib.pyplot as plt
from IPython import display

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
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
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


