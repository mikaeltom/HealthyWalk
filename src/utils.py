"""
Mikael Tom
9 June 2026
File that contains the functions necessary for computing
the learning curves and display the learning curves.
"""
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import enum


class DisplayType(enum.Enum):
    """
    Class used only for display behaviours.
    The idea is to indicate the mode the user selected.
    """
    NO_DISPLAY = 0 # No GUI, just the learning curve displayed
    SIMPLE_DISPLAY = 1 # GUI showing the map and agent actions
    RATING_DISPLAY = 2 # GUI allowing to rate the agent paths


class LogReward(BaseCallback):
    """
    This class overwrite the BaseCallback. The idea is to track all rewards to display a learning curve
    because the method of using  gym.wrappers.RecordEpisodeStatistics(env) in WPO takes only the
    last 100 episodes, but we have more than that.
    This is inspired by :
    https://medium.com/@bravekjh/building-an-ai-agent-with-reinforcement-learning-and-human-in-the-loop-labeling-c8e16d239a33
    https://stable-baselines3.readthedocs.io/en/md-doc/guide/examples.html
    https://www.reddit.com/r/reinforcementlearning/comments/1hbt64c/how_to_dynamically_modify_hyperparameters_during/
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0] # source : https://medium.com/@bravekjh/building-an-ai-agent-with-reinforcement-learning-and-human-in-the-loop-labeling-c8e16d239a33
        if self.locals["dones"][0]: # source : https://www.reddit.com/r/reinforcementlearning/comments/1kkejga/pettingzoo_has_anyone_managed_to_get_logs_in_sb3/
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True

def display_learning_curve(log_reward):
    """
    Function that receives the rewards from the episodes and displays the learning curve.
    For visualisation purposes, the curves was smoothed in order to avoid having too
    many fluctuations. It is smoothed using Moving Avrage Filter : https://en.wikipedia.org/wiki/Moving_average
    using the convolve function.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    rewards = log_reward.episode_rewards
    smooth_curve = np.convolve(rewards, np.ones(500),
                               'valid') / 500  # source : https://stackoverflow.com/questions/68519785/how-to-smooth-accuracy-and-loss-curves-in-deep-learning-models
    ax.plot(smooth_curve)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rewards")
    ax.set_title("Learning Curve (smoothed) during the training of Healthy Walk agent")
    return fig