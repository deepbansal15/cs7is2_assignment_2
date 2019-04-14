from spinup import vpg
import tensorflow as tf
import roboschool
import gym

env_fn = lambda : gym.make('RoboschoolAnt-v1')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='data/vpg_bench/seed20', exp_name='vpg_ant')

vpg(env_fn=env_fn, ac_kwargs=ac_kwargs,seed=20, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)

#test on seed 10,20

# python spinningup/spinup/utils/plot.py src/data/ppo_bench
