import tensorflow as tf

from retro_contest.local import make

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer, StackedBoxSpace

import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env
import numpy as np


print('creating env')
z = StackedBoxSpace(np.zeros((84,84,1)), 4)

env = AllowBacktracking(make_env(stack=False, scale_rew=False))

env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

#print(env.action_space.n)
#StackedBox(84,84,1)

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

print('starting tf session')

with tf.Session(config=config) as sess:

    print('creating agent')

    online_net, target_net = rainbow_models(sess, env.action_space.n,
                              gym_space_vectorizer(env.observation_space),
                              min_val=-200,
                              max_val=200)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, "saved_models/rainbow1.ckpt")

    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)

    print('model loaded')

    # RESET
    env.reset_start()
    obs = env.reset_wait()
    #print(obs1[0][0].shape)

    #print(online_net.step(obs1, obs1))

    for i in range(50):

        action = online_net.step(obs, obs)['actions'][0]

        env.step_start([action])

        obs, reward, _, _, = env.step_wait()

        print(action, reward)



    #env.step_wait()





'''
import pickle

with open('saved_frame.p', 'wb') as f:

    pickle.dump(obs1[0][0], f)
'''



    #online_net.step(obs, obs)

    # need to get actions from model
    # need to advance the environemtn with the action

    # fit into the framework of simple-agent.py

    # don't really need to worry about the env since it'll be handled remotely..
    # but still, to assess locally, need to sort it out.
