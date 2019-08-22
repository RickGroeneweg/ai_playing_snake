import gym
from gym.envs.registration import register

# delete if it's registered
env_name = 'snake-v0'
#if env_name in gym.envs.registry.env_specs:
#    del gym.envs.registry.env_specs[env_name]

register(
    id=env_name,
    entry_point='gym_snake.envs:SnakeEnv'
)
