import gym
# from gym import logger

gym.logger.setLevel(gym.logger.DISABLED)

env = gym.make("CartPole-v1", new_step_api=False)
