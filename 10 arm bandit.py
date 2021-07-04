import numpy as np
from numpy import random
import matplotlib.pyplot as plt

np.random.seed(42)
actual = np.random.normal(0, 1, 10)
agent = np.random.normal(0, 1e-6, 10)
agent_orig = agent.copy()
reward_avg = []
reward_avg.append(0)

epsilon = .75
tasks = 50000

for i in range(1, tasks):
    alpha = 1  # previously was 1/i
    # setting the policy here
    if random.random() < epsilon:  # 0.1 probability that same action will be chosen. not worth fixing
        action = np.argmax(agent)
    else:
        action = np.random.choice(len(agent))
    # temporal difference calculation
    reward = actual[action] + np.random.normal(0, 0.1)
    new_estimate = agent[action] + (reward - agent[action]) * alpha
    agent[action] = new_estimate
    # tracking rewards
    reward_avg.append((reward + sum(reward_avg))/i)

actual, agent
range(tasks)

plt.plot(reward_avg)

np.argmax(agent)

# This was fun
# The agent usually finds a reward it can trust and then sticks with it through all cycles
# Can tell by companing the actual vs agent representations haha
# changing alpha changes this a lot
# If had more time/computing power would love to run multiple agents and plot them as an average
